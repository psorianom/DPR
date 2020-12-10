"""
This fast script transforms the output checkpoint file from DPR code to a HuggingFace's Roberta model
Based on DPR's own genrate_dense_embeddings and dense_retriever.py, and transformer's converter scripts.
It aims camembert/roberta models!
"""
from dpr.models import init_biencoder_components
from dpr.options import set_encoder_params_from_state, print_args

from dpr.utils.model_utils import get_model_obj, load_states_from_checkpoint

import argparse
import pathlib

import torch
from fairseq.models.roberta.model import RobertaModel as FairseqRobertaModel
from fairseq.modules import TransformerSentenceEncoderLayer

from transformers.modeling_bert import BertIntermediate, BertLayer, BertOutput, BertSelfAttention, BertSelfOutput
from transformers.modeling_roberta import RobertaConfig, RobertaForMaskedLM, RobertaForSequenceClassification
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = "Hello world! cécé herlolip"

def convert_roberta_checkpoint_to_pytorch(roberta: FairseqRobertaModel, pytorch_dump_folder_path: str, classification_head: bool
):
    """
    Copy/paste/tweak roberta's weights to our BERT structure.
    """

    roberta.eval()  # disable dropout
    roberta_sent_encoder = roberta.fairseq_roberta.model.encoder.sentence_encoder
    config = RobertaConfig(
        vocab_size=roberta_sent_encoder.embed_tokens.num_embeddings,
        hidden_size=roberta.fairseq_roberta.args.encoder_embed_dim,
        num_hidden_layers=roberta.fairseq_roberta.args.encoder_layers,
        num_attention_heads=roberta.fairseq_roberta.args.encoder_attention_heads,
        intermediate_size=roberta.fairseq_roberta.args.encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
    )
    if classification_head:
        config.num_labels = roberta.model.classification_heads["mnli"].out_proj.weight.shape[0]
    print("Our BERT config:", config)

    model = RobertaForSequenceClassification(config) if classification_head else RobertaForMaskedLM(config)
    model.eval()

    # Now let's copy all the weights.
    # Embeddings
    model.roberta.embeddings.word_embeddings.weight = roberta_sent_encoder.embed_tokens.weight
    model.roberta.embeddings.position_embeddings.weight = roberta_sent_encoder.embed_positions.weight
    model.roberta.embeddings.token_type_embeddings.weight.data = torch.zeros_like(
        model.roberta.embeddings.token_type_embeddings.weight
    )  # just zero them out b/c RoBERTa doesn't use them.
    model.roberta.embeddings.LayerNorm.weight = roberta_sent_encoder.emb_layer_norm.weight
    model.roberta.embeddings.LayerNorm.bias = roberta_sent_encoder.emb_layer_norm.bias

    for i in range(config.num_hidden_layers):
        # Encoder: start of layer
        layer: BertLayer = model.roberta.encoder.layer[i]
        roberta_layer: TransformerSentenceEncoderLayer = roberta_sent_encoder.layers[i]

        # self attention
        self_attn: BertSelfAttention = layer.attention.self
        assert (
            roberta_layer.self_attn.k_proj.weight.data.shape
            == roberta_layer.self_attn.q_proj.weight.data.shape
            == roberta_layer.self_attn.v_proj.weight.data.shape
            == torch.Size((config.hidden_size, config.hidden_size))
        )

        self_attn.query.weight.data = roberta_layer.self_attn.q_proj.weight
        self_attn.query.bias.data = roberta_layer.self_attn.q_proj.bias
        self_attn.key.weight.data = roberta_layer.self_attn.k_proj.weight
        self_attn.key.bias.data = roberta_layer.self_attn.k_proj.bias
        self_attn.value.weight.data = roberta_layer.self_attn.v_proj.weight
        self_attn.value.bias.data = roberta_layer.self_attn.v_proj.bias

        # self-attention output
        self_output: BertSelfOutput = layer.attention.output
        assert self_output.dense.weight.shape == roberta_layer.self_attn.out_proj.weight.shape
        self_output.dense.weight = roberta_layer.self_attn.out_proj.weight
        self_output.dense.bias = roberta_layer.self_attn.out_proj.bias
        self_output.LayerNorm.weight = roberta_layer.self_attn_layer_norm.weight
        self_output.LayerNorm.bias = roberta_layer.self_attn_layer_norm.bias

        # intermediate
        intermediate: BertIntermediate = layer.intermediate
        assert intermediate.dense.weight.shape == roberta_layer.fc1.weight.shape
        intermediate.dense.weight = roberta_layer.fc1.weight
        intermediate.dense.bias = roberta_layer.fc1.bias

        # output
        bert_output: BertOutput = layer.output
        assert bert_output.dense.weight.shape == roberta_layer.fc2.weight.shape
        bert_output.dense.weight = roberta_layer.fc2.weight
        bert_output.dense.bias = roberta_layer.fc2.bias
        bert_output.LayerNorm.weight = roberta_layer.final_layer_norm.weight
        bert_output.LayerNorm.bias = roberta_layer.final_layer_norm.bias
        # end of layer

    if classification_head:
        model.classifier.dense.weight = roberta.model.classification_heads["mnli"].dense.weight
        model.classifier.dense.bias = roberta.model.classification_heads["mnli"].dense.bias
        model.classifier.out_proj.weight = roberta.model.classification_heads["mnli"].out_proj.weight
        model.classifier.out_proj.bias = roberta.model.classification_heads["mnli"].out_proj.bias
    else:
        # LM Head
        model.lm_head.dense.weight = roberta.fairseq_roberta.model.encoder.lm_head.dense.weight
        model.lm_head.dense.bias = roberta.fairseq_roberta.model.encoder.lm_head.dense.bias
        model.lm_head.layer_norm.weight = roberta.fairseq_roberta.model.encoder.lm_head.layer_norm.weight
        model.lm_head.layer_norm.bias = roberta.fairseq_roberta.model.encoder.lm_head.layer_norm.bias
        model.lm_head.decoder.weight = roberta.fairseq_roberta.model.encoder.lm_head.weight
        model.lm_head.decoder.bias = roberta.fairseq_roberta.model.encoder.lm_head.bias

    # Let's check that we get the same results.
    input_ids: torch.Tensor = roberta.fairseq_roberta.encode(SAMPLE_TEXT).unsqueeze(0)  # batch of size 1

    our_output = model(input_ids)[0]
    if classification_head:
        their_output = roberta.fairseq_roberta.model.classification_heads["mnli"](roberta.extract_features(input_ids))
    else:
        their_output = roberta.fairseq_roberta.model(input_ids)[0]
    print(our_output.shape, their_output.shape)
    max_absolute_diff = torch.max(torch.abs(our_output - their_output)).item()
    print(f"max_absolute_diff = {max_absolute_diff}")  # ~ 1e-7
    success = torch.allclose(our_output, their_output, atol=1e-3)
    print("Do both models output the same tensors?", "🔥" if success else "💩")
    if not success:
        raise Exception("Something went wRoNg")

    pathlib.Path(pytorch_dump_folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Saving model to {pytorch_dump_folder_path}")
    model.save_pretrained(pytorch_dump_folder_path)


def load_bi_encoders(dpr_cp_path: str, args):

    def load_saved_state_into_model(encoder, prefix="ctx_model."):
        encoder.eval()
        model_to_load = get_model_obj(encoder)
        prefix_len = len(prefix)
        encoder_state = {key[prefix_len:]: value for (key, value) in saved_state.model_dict.items() if
                     key.startswith(prefix)}
        model_to_load.load_state_dict(encoder_state)
        return model_to_load

    saved_state = load_states_from_checkpoint(dpr_cp_path)
    set_encoder_params_from_state(saved_state.encoder_params, args)
    print(args)
    tensorizer, encoder, _ = init_biencoder_components(args.encoder_model_type, args, inference_only=True)
    encoder_ctx = encoder.ctx_model
    encoder_question = encoder.question_model

    final_encoder_ctx = load_saved_state_into_model(encoder_ctx, "ctx_model.")
    final_encoder_question = load_saved_state_into_model(encoder_question, "question_model.")

    return final_encoder_ctx, final_encoder_question


parser = argparse.ArgumentParser()
parser.add_argument("--arch", default="roberta_base")
parser.add_argument("--task", default="language_modeling")
args = parser.parse_args()
DPR_CP_PATH = "./dpr_biencoder.34.708"
OUTPUT_CTX_ENCODER ="./encoder_ctx"
OUTPUT_QUESTION_ENCODER = "./encoder_question"

final_encoder_ctx, final_encoder_question = load_bi_encoders(DPR_CP_PATH, args)

convert_roberta_checkpoint_to_pytorch(final_encoder_ctx, OUTPUT_CTX_ENCODER, False)
convert_roberta_checkpoint_to_pytorch(final_encoder_question, OUTPUT_QUESTION_ENCODER, False)
