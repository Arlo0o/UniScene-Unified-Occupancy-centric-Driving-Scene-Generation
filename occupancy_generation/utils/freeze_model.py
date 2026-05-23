import torch
from mmengine import MMLogger
logger = MMLogger.get_instance('genocc')
import torch.distributed as dist

def freeze_model(model, freeze_dict):
    # given a model and a dictionary of booleans, freeze the model
    # according to the dictionary
    for key in freeze_dict:
        if freeze_dict[key]:
            for param in getattr(model, key).parameters():
                param.requires_grad = False
            logger = MMLogger.get_current_instance()
            logger.info(f'Freezed {key} parameters')
                
if __name__ == '__main__':
    model = torch.nn.Sequential(
        torch.nn.Linear(1, 1),
        torch.nn.Linear(1, 1),
        torch.nn.Linear(1, 1)
    )
    print(model)
    freeze_dict = {'0': True, '1': False, '2': True}
    freeze_model(model, freeze_dict)
    import pdb; pdb.set_trace()