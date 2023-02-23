import torch
from aesthetic_predictor.simple_inference import AestheticPredictor, normalized


class Gradient_Ascent:

    def __init__(self):
        self.epsilon = 0.25
        self.num_steps = 5
        self.alpha = 0.005
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.y_target = torch.tensor([9.0]).to(self.device).view(1, 1)
        self.aestethetic_pred = AestheticPredictor()
        self.mlp = self.aestethetic_pred.mlp
        self.mse_loss = torch.nn.MSELoss()
        self.model_params = self.aestethetic_pred.get_model_params()
        self.optimizer = self.aestethetic_pred.get_optimizer()

    def fgsm(self, embedding, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the predict_aesthetic_score image
        perturbed_image = embedding - self.alpha * sign_data_grad
        # Return the perturbed image
        return perturbed_image

    def get_gradient(self, input, text_input = False, image_input = True):
        #image_features = self.aestethetic_pred.pil_to_features(predict_aesthetic_score)
        features = self.aestethetic_pred.get_features(input, text_input, image_input)
        features.requires_grad = True
        output = self.mlp(features)
        print(output)
        #loss = self.mse_loss(output, self.y_target)
        loss = -output
        self.mlp.zero_grad()
        loss.backward()
        data_grad = features.grad.data
        #print(data_grad)
        perturbed_image_features = self.fgsm(features, data_grad)
        pred = self.mlp(perturbed_image_features)
        print(pred)
        return perturbed_image_features

    def get_feature_normalization(self, input):
        return normalized(input.cpu().detach().numpy(), return_l2=True)





