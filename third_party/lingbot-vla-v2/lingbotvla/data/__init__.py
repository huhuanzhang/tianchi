# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from importlib import import_module

# Import the requested data API without eagerly importing all dataset backends.
# The native clean reader only needs vla_data.utils.FeatureTransform.
_MODULES = {
    "build_chat_template": ".chat_template",
    "build_dataloader": ".data_loader",
    **dict.fromkeys(["CollatePipeline", "DataCollatorWithPacking", "DataCollatorWithPadding",
                     "DataCollatorWithPositionIDs", "MakeMicroBatchCollator",
                     "TextSequenceShardCollator", "UnpackDataCollator"], ".data_collator"),
    **dict.fromkeys(["build_iterative_dataset", "build_mapping_dataset", "build_vla_dataset"], ".dataset"),
    **dict.fromkeys(["OmniDataCollatorWithPacking", "OmniDataCollatorWithPadding",
                     "OmniSequenceShardCollator", "VLADataCollatorWithPacking"], ".multimodal.data_collator"),
    "build_multimodal_chat_template": ".multimodal.multimodal_chat_template",
}
__all__ = list(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(_MODULES[name], __name__), name)
    globals()[name] = value
    return value
