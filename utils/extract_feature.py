import torch

def extract_feature(model, loader, use_gpu=True):
    features = torch.FloatTensor()
    image_paths = []
    with torch.no_grad():
        for (inputs, labels) in loader:
            if use_gpu:
                input_img = inputs.to('cuda')
            else:
                input_img = inputs
            outputs = model(input_img)
            f1 = outputs[0].data.cpu()
            # flip
            inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1))
            if use_gpu:
                input_img = inputs.to('cuda')
            else:
                input_img = inputs
            outputs = model(input_img)
            f2 = outputs[0].data.cpu()
            ff = f1 + f2

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))
            features = torch.cat((features, ff), 0)
            image_paths.extend(labels)
    return features, image_paths
