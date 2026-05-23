import pickle
from collections import defaultdict

# 指定原始 .pkl 文件的路径
file_path = '/lpai/volumes/ad-lmm-data-proc-bd-ga/hzhu/data/nuplan_pkls/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_val_noted_with_ego_stationary_dynamic_others.pkl'

# 指定新的 .pkl 文件的路径
new_file_path = '/lpai/volumes/ad-lmm-data-proc-bd-ga/hzhu/data/nuplan_pkls/nuplan_pkls/nuplan_10hz_pkl/mini/nuplan_mini_10hz_val_noted_with_ego_stationary_dynamic_others_v2.pkl'

# 读取原始 .pkl 文件
with open(file_path, 'rb') as file:
    data3 = pickle.load(file)

# 初始化一个字典来存储每个场景的所有 token
scene_tokens_dict = defaultdict(list)

# 遍历 data3['infos'] 中的每个条目
for info in data3['infos']:
    # 获取当前条目的 scene_token
    # scene_token = info['scene_token']
    scene_token = info['scene_token']
    # 获取当前条目的 token
    token = info['token']
    # 将 token 添加到对应的 scene_token 列表中
    scene_tokens_dict[scene_token].append(token)

# 将 scene_tokens_dict 转换为一个列表，每个元素是一个场景的 token 列表
scene_tokens = list(scene_tokens_dict.values())

# 将 scene_tokens 添加到 data3 字典中
data3['scene_tokens'] = scene_tokens

# 将更新后的 data3 保存为新的 .pkl 文件
with open(new_file_path, 'wb') as file:
    pickle.dump(data3, file)

print(f"New pickle file with 'scene_tokens' has been saved to {new_file_path}")