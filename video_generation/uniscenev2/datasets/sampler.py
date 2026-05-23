from collections import OrderedDict, defaultdict
from typing import Iterator, List, Optional
import logging
from pprint import pformat
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler

from uniscenev2.utils.misc import format_numel_str, get_logger



# use pandarallel to accelerate bucket processing
# NOTE: pandarallel should only access local variables
def apply(data, method=None, frame_interval=None, seed=None, num_bucket=None):
    return method(
        data["num_frames"],
        data["height"],
        data["width"],
        frame_interval,
        seed + data["id"] * num_bucket,
    )


class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def reset(self) -> None:
        self.start_index = 0

    def state_dict(self, step) -> dict:
        return {"start_index": step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)



class NuScenesVariableBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset,
        bucket_config: dict,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.bs_config = bucket_config
        self.verbose = verbose
        self.last_micro_batch_access_index = 0

        self._default_bucket_sample_dict = self.dataset.as_buckets()
        self._default_bucket_micro_batch_count = OrderedDict()
        self.approximate_num_batch = 0
        # process the samples
        for bucket_id, data_list in self._default_bucket_sample_dict.items():
            # handle droplast
            bs_per_gpu = self.bs_config[bucket_id]
            if bs_per_gpu == -1:
                logging.warning(f"Got bs=-1, we drop {bucket_id}.")
                continue
            remainder = len(data_list) % bs_per_gpu

            if remainder > 0:
                if not self.drop_last:
                    # if there is remainder, we pad to make it divisible
                    data_list += data_list[: bs_per_gpu - remainder]
                else:
                    # we just drop the remainder to make it divisible
                    data_list = data_list[:-remainder]
            self._default_bucket_sample_dict[bucket_id] = data_list
            # compute how many micro-batches each bucket has
            num_micro_batches = len(data_list) // bs_per_gpu
            self._default_bucket_micro_batch_count[bucket_id] = num_micro_batches
            self.approximate_num_batch += num_micro_batches
        self._print_bucket_info(self._default_bucket_sample_dict)

    def __iter__(self) -> Iterator[List[str]]:
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_last_consumed = OrderedDict()

        bucket_sample_dict = OrderedDict({k: v for k, v in self._default_bucket_sample_dict.items()})
        for bucket_id, data_list in bucket_sample_dict.items():
            # handle shuffle
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in self._default_bucket_micro_batch_count.items():
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # make the number of bucket accesses divisible by dp size
        remainder = len(bucket_id_access_order) % self.num_replicas
        if remainder > 0:
            if self.drop_last:
                bucket_id_access_order = bucket_id_access_order[: len(bucket_id_access_order) - remainder]
            else:
                bucket_id_access_order += bucket_id_access_order[: self.num_replicas - remainder]

        # prepare each batch from its bucket
        # according to the predefined bucket access order
        num_iters = len(bucket_id_access_order) // self.num_replicas
        # NOTE: all the dict/indexes should be the same across all devices.
     
        start_iter_idx = self.last_micro_batch_access_index // self.num_replicas

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        self.last_micro_batch_access_index = start_iter_idx * self.num_replicas
        # print(self.last_micro_batch_access_index,"--------bucket_id_access_order: ----",  len(bucket_id_access_order),  )
        for i in range(self.last_micro_batch_access_index):
            
            bucket_id = bucket_id_access_order[i]
            bucket_bs = self.bs_config[bucket_id]
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

        for i in range(start_iter_idx, num_iters):
            bucket_access_list = bucket_id_access_order[i * self.num_replicas: (i + 1) * self.num_replicas]
            self.last_micro_batch_access_index += self.num_replicas

            # compute the data samples consumed by each access
            bucket_access_boundaries = []
            for bucket_id in bucket_access_list:
                bucket_bs = self.bs_config[bucket_id]
                last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
                bucket_access_boundaries.append([last_consumed_index, last_consumed_index + bucket_bs])

                # update consumption
                if bucket_id in bucket_last_consumed:
                    bucket_last_consumed[bucket_id] += bucket_bs
                else:
                    bucket_last_consumed[bucket_id] = bucket_bs

            # compute the range of data accessed by each GPU
            bucket_id = bucket_access_list[self.rank]
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0]: boundary[1]]

            # encode t, h, w into the sample index
            cur_micro_batch = [f"{idx}-{bucket_id}" for idx in cur_micro_batch]
            yield cur_micro_batch

        self.reset()

    def __len__(self) -> int:
        return self.get_num_batch() // self.num_replicas

    def reset(self):
        self.last_micro_batch_access_index = 0

    def get_num_batch(self) -> int:
        # calculate the number of batches
        if self.verbose:
            self._print_bucket_info(self._default_bucket_sample_dict)
        return self.approximate_num_batch

    def _print_bucket_info(self, bucket_sample_dict: dict) -> None:
        # collect statistics
        total_samples = 0
        total_batch = 0
        full_dict = defaultdict(lambda: [0, 0])
        num_img_dict = defaultdict(lambda: [0, 0])
        num_vid_dict = defaultdict(lambda: [0, 0])
        for k, v in bucket_sample_dict.items():
            size = len(v)
            real_h, real_w, fps = map(int, k.split("-")[:-1])
            real_t = k.split("-")[-1]
            real_t = real_t if real_t == "full" else int(real_t)
            num_batch = size // self.bs_config[k]

            total_samples += size
            total_batch += num_batch

            full_dict[k][0] += size
            full_dict[k][1] += num_batch

            if real_t == 1:
                num_img_dict[k][0] += size
                num_img_dict[k][1] += num_batch
            else:
                num_vid_dict[k][0] += size
                num_vid_dict[k][1] += num_batch

        # log
        if dist.get_rank() == 0:
            logging.info("Bucket Info:")
            logging.info(
                "Bucket [#sample, #batch] by aspect ratio:\n%s", pformat(full_dict, sort_dicts=False)
            )
            logging.info(
                "Image Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_img_dict, sort_dicts=False)
            )
            logging.info(
                "Video Bucket [#sample, #batch] by HxWxT:\n%s", pformat(num_vid_dict, sort_dicts=False)
            )
            logging.info(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                format_numel_str(total_batch),
                format_numel_str(total_samples),
                len(bucket_sample_dict),
            )

    def state_dict(self, num_steps: int) -> dict:
        # the last_micro_batch_access_index in the __iter__ is often
        # not accurate during multi-workers and data prefetching
        # thus, we need the user to pass the actual steps which have been executed
        # to calculate the correct last_micro_batch_access_index
        return {
            "seed": self.seed,
            "epoch": self.epoch,
            "last_micro_batch_access_index": num_steps * self.num_replicas,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)
