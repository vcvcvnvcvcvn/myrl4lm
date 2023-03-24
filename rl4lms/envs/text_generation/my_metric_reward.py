from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import PreTrainedModel
from abc import ABC, abstractclassmethod
import torch
from typing import List, Dict, Tuple, Any
from abc import abstractmethod
import numpy as np
from torch import nn
import os
from rl4lms.envs.text_generation.observation import Observation
from transformers import AutoModelForCausalLM, AutoTokenizer

REWARD_CHECKPOINT_PATH = "reward_model/rm_checkpoint/pytorch_model.bin"
if not os.path.exists(REWARD_CHECKPOINT_PATH):
    os.makedirs("reward_model/rm_checkpoint", exist_ok=True)
    os.system(
        f"wget -O {REWARD_CHECKPOINT_PATH} \
        https://huggingface.co/Jieming/gpt2_reward_model/resolve/main/pytorch_model.bin"
    )
SFT_MODEL_PATH = "gavin124/gpt2-finetuned-cnn-summarization-v2"
MAX_LENGTH = 550

class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path)
        self.config = model.config
        # `gpt-neo(x)` models use `hidden_size` attribute names instead of `n_embd``
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.transformer = model.transformer
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False)###加一个输出层，输出一个分数
        self.tokenizer = AutoTokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")###load the tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token##pad token is equal to eos_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]##?

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        loss = None
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hidden_states = transformer_outputs[0]

        rewards = self.v_head(hidden_states).squeeze(-1)
        chosen_end_scores = []
        rejected_end_scores = []

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        chosen = input_ids[:bs]
        rejected = input_ids[bs:]
        chosen_rewards = rewards[:bs]
        rejected_rewards = rewards[bs:]

        # Compute pairwise loss. Only backprop on the last value before padding
        loss = 0
        inference = False
        for i in range(bs):
            if torch.all(torch.eq(chosen[i], rejected[i])).item():
                c_inds = (chosen[i] == self.PAD_ID).nonzero()
                c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
                chosen_end_scores.append(chosen_rewards[i, c_ind - 1])
                inference = True
                continue

            # Check if there is any padding otherwise take length of sequence
            c_inds = (chosen[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else chosen.shape[1]
            r_inds = (rejected[i] == self.PAD_ID).nonzero()
            r_ind = r_inds[0].item() if len(r_inds) > 0 else rejected.shape[1]
            end_ind = max(c_ind, r_ind)

            # Retrieve first index where trajectories diverge
            divergence_ind = (chosen[i] != rejected[i]).nonzero()[0]
            assert divergence_ind > 0

            # Index into the correct rewards
            c_truncated_reward = chosen_rewards[i][divergence_ind:end_ind]
            r_truncated_reward = rejected_rewards[i][divergence_ind:end_ind]

            # Append the last rewards to the list of end scores
            chosen_end_scores.append(c_truncated_reward[-1])
            rejected_end_scores.append(r_truncated_reward[-1])

            # Compute loss
            loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()
        loss = loss / bs

        if not inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            rejected_end_scores = torch.stack(rejected_end_scores)

        if inference:
            chosen_end_scores = torch.stack(chosen_end_scores)
            return {"chosen_end_scores": chosen_end_scores}

        return {
            "loss": loss,
            "chosen_end_scores": chosen_end_scores,
            "rejected_end_scores": rejected_end_scores,
        }

# print('Load the pre-trained model')
# # Load the pre-trained reward model
# rw_tokenizer = AutoTokenizer.from_pretrained("gavin124/gpt2-finetuned-cnn-summarization-v2")
# rw_tokenizer.pad_token = rw_tokenizer.eos_token
# rw_model = GPTRewardModel(SFT_MODEL_PATH)
# rw_model.load_state_dict(torch.load(REWARD_CHECKPOINT_PATH))
# rw_model.half()
# rw_model.eval()
# rw_device = torch.device("cuda:{}".format(0))  # set reward model device
# rw_model.to(rw_device)
print('Do not load the premodel')
rw_model = None
rw_tokenizer = None
rw_device = None

def get_scores(samples: List[str]):
    scores_list = []
    batch_size = 2
    for i in range(0, len(samples), batch_size):
        sub_samples = samples[i : i + batch_size]
        sub_samples = ["<|startoftext|>" + chosen + "<|endoftext|>" for chosen in sub_samples]
        encodings_dict = rw_tokenizer(
            sub_samples,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings_dict["input_ids"].to(rw_device)
        attn_masks = encodings_dict["attention_mask"].to(rw_device)
        input_ids = input_ids.repeat(2, 1)
        attn_masks = attn_masks.repeat(2, 1)
        with torch.no_grad():
            sub_scores = rw_model(input_ids=input_ids, attention_mask=attn_masks)
        scores_list.append(sub_scores["chosen_end_scores"])
    scores = torch.cat(scores_list, dim=0)
    return scores

def reward_fn(prompt_texts,generated_texts,reference_texts):
    ###format_step
    samples = [text[0]+ "TL;DR: "+text[1] for text in zip(prompt_texts,generated_texts)]
    original_samples = [text[0]+ "TL;DR: "+text[1][0] for text in zip(prompt_texts,reference_texts)]
    #original_samples = [text + post_summary_dict[text.strip()] for text in original_samples]
    #original_samples = [text + refrences for text in original_samples]
    original_scores = get_scores(original_samples)
    scores = get_scores(samples)
    norms_scores = scores - original_scores####compare with original summary 
    return torch.mean(norms_scores).item()


class RewardFunction(ABC):
    @abstractclassmethod
    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        """
        Callable for reward functions for text generation

        Args:
            current_observation (Observation): previous observation (s)
            action (int): action performed (a) at s
            next_observation (Observation): observation after the action was performed (s')
            done (bool): whether the episode is finished or not
            meta_info (dict) - other information regarding textual sample
        Returns:
            float: scalar reward
        """
        raise NotImplementedError


class BaseMetric:
    @abstractmethod
    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ):
        """
        Returns a dict where key is the metric name and value is again a dict consisting of tuple of individual scores (if any) and corpus level score

        eg. {
            metric_name: (individual_scores, corpus_level_score)
            "metric_1": ([0.5, 0.5, 0.8], 0.1)
        }

        """
        raise NotImplementedError
    

class myRewardMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()

    def compute(
        self,
        prompt_texts: List[str],
        generated_texts: List[str],
        reference_texts: List[List[str]],
        meta_infos: List[Dict[str, Any]] = None,
        model: PreTrainedModel = None,
        split_name: str = None,
    ) -> Tuple[List[float], float]:
        with torch.no_grad():
            # metric_results = self._metric.compute(
            #     predictions=generated_texts,
            #     references=reference_texts,
            #     lang=self._language,
            #     device=self._last_gpu,
            # )
            scores = reward_fn(prompt_texts,generated_texts,reference_texts)
            metric_dict = {"semantic/my_reward_score": (None,scores)}
            return metric_dict
        

class myRewardFunction(RewardFunction):
    def __init__(self) -> None:
        super().__init__()
        self._metric = myRewardMetric()
    def __call__(
        self,
        current_observation: Observation,
        action: int,
        next_observation: Observation,
        done: bool,
        meta_info: Dict[str, Any] = None,
    ) -> float:
        if done:
            prompts = [next_observation.prompt_or_input_text]
            references = [next_observation.target_or_reference_texts]
            predicted = [next_observation.context_text]
            metric_results = self._metric.compute(prompts, predicted, references)
            my_score = metric_results["semantic/my_reward_score"][1]
            return my_score
        return 0