import torch
from torchvision import utils
import cv2
import numpy as np
import os
import torch.nn as nn

from networks_and_related_funs.ClassicNetwork.ResNet import ResNet50
from feature_extractor import feature_config


class FeatureExtractor:
	""" Class for extracting activations and 
	registering gradients from targetted intermediate layers
	"""
	def __init__(self, model, target_layers):
		self.model = model
		self.target_layers = target_layers
		self.gradients = []

	def save_gradient(self, grad):
		self.gradients.append(grad)

	def __call__(self, x):
		outputs = []
		self.gradients = []
		for name, module in self.model._modules.items():
			x = module(x)
			if name in self.target_layers:
				x.register_hook(self.save_gradient)  # 注册回调钩子
				outputs += [x]
		return outputs, x


class ModelOutputs:
	""" Class for making a forward pass, and getting:
	1. The network output. 模型输出
	2. Activations from intermeddiate targetted layers. 中间的目标层的激活值
	3. Gradients from intermeddiate targetted layers. 中间的目标层的梯度值
	"""
	def __init__(self, model, target_layers, use_cuda):
		self.model = model
		self.feature_extractor = FeatureExtractor(self.model, target_layers)
		self.cuda = use_cuda

	def get_gradients(self):
		return self.feature_extractor.gradients

	def __call__(self, x):
		target_activations, output = self.feature_extractor(x)
		output = output.view(output.size(0), -1)  # 展平目标层的输出
		if self.cuda:
			output = model.fc(output)
		else:
			output = model.fc(output)
		return target_activations, output


def preprocess_image(img):  # 预处理图片以便输入模型
	# TODO:这里根据实际的数据集进行调整
	means=[0.485, 0.456, 0.406]
	stds=[0.229, 0.224, 0.225]

	preprocessed_img = img.copy()[:, :, ::-1] # BGR -> RGB
	for i in range(3): # 标准化
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
		preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
	preprocessed_img = \
		np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
	preprocessed_img = torch.from_numpy(preprocessed_img)
	preprocessed_img.unsqueeze_(0)
	input = preprocessed_img
	input.requires_grad = True
	return input


def show_cam_on_image(img, mask, name, path_dir):  # 融合热力图和原图并保存
	heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
	heatmap = np.float32(heatmap) / 255
	cam = heatmap + np.float32(img)
	cam = cam / np.max(cam)
	cv2.imwrite(os.path.join(path_dir, "cam_" + name + '.png'), np.uint8(255 * cam))


class GradCam:
	def __init__(self, model, target_layer_names, use_cuda):
		self.model = model
		self.model.eval()  # 模型进入验证模式
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.to(device)
		self.extractor = ModelOutputs(self.model, target_layer_names, use_cuda)

	def forward(self, inputs):
		return self.model(inputs)

	def __call__(self, inputs, index=None):  # index表示从模型的第几个分类往回求热力图，不写就默认使用模型预测分类
		if self.cuda:
			features, output = self.extractor(inputs.to(device))
		else:
			features, output = self.extractor(inputs)
		if index is None:
			index = np.argmax(output.cpu().data.numpy())

		# 构造一个类似于这样的one_hot向量 [0,0,0,1,0]
		one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
		one_hot[0][index] = 1
		one_hot = torch.Tensor(torch.from_numpy(one_hot))
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.to(device) * output)
		else:
			one_hot = torch.sum(one_hot * output)

		self.model.zero_grad()  # features和classifier不包含，可以重新加回去试一试，会报错不包含这个对象。
		one_hot.backward(retain_graph=True)  # 这里适配我们的torch0.4及以上，我用的1.0也可以完美兼容。（variable改成graph即可）

		grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()
		target = features[-1]
		target = target.cpu().data.numpy()[0, :]

		weights = np.mean(grads_val, axis=(2, 3))[0, :]
		cam = np.zeros(target.shape[1:], dtype=np.float32)
		for i, w in enumerate(weights):
			cam += w * target[i, :, :]

		cam = np.maximum(cam, 0)
		cam = cv2.resize(cam, (224, 224))
		cam = cam - np.min(cam)
		cam = cam / np.max(cam)
		return cam


class GuidedBackpropReLUModel:
	def __init__(self, model, use_cuda):
		self.model = model
		self.model.eval()
		self.cuda = use_cuda
		if self.cuda:
			self.model = model.to(device)
		for module in self.model.named_modules():
			module[1].register_backward_hook(self.bp_relu)

	def bp_relu(self, module, grad_in, grad_out):
		if isinstance(module, nn.ReLU):
			return torch.clamp(grad_in[0], min=0.0)

	def forward(self, input):
		return self.model(input)

	def __call__(self, input, index=None):
		if self.cuda:
			output = self.forward(input.to(device))
		else:
			output = self.forward(input)
		output = output[1]  # 换了CSER_ResNet框架出来的模型
		if index == None:
			index = np.argmax(output.cpu().data.numpy())
		one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
		one_hot[0][index] = 1
		one_hot = torch.from_numpy(one_hot)
		one_hot.requires_grad = True
		if self.cuda:
			one_hot = torch.sum(one_hot.to(device) * output)
		else:
			one_hot = torch.sum(one_hot * output)
		one_hot.backward(retain_graph=True)  # 反向向前传导到整个图（one_hot刚才和output联系起来了）
		output = input.grad.cpu().data.numpy()
		output = output[0, :, :, :]
		return output


device = torch.device("cuda:{}".format(feature_config.cuda_num) if torch.cuda.is_available() else "cpu")
resnet = ResNet50(num_classes=feature_config.class_number).to(device)

if __name__ == '__main__':
	""" python grad_cam.py <path_to_image>
	1. Loads an image with opencv.
	2. Preprocesses it for VGG19 and converts to a pytorch variable.
	3. Makes a forward pass to find the category index with the highest score,
	and computes intermediate activations.
	Makes the visualization. """

	MODEL_PATH = "Model_and_Log/ResNet/47_80/model.pkl"
	dir = 'cam'
	img_path = 'cam/raw'
	save_path = 'cam/new'
	path_list = [dir, img_path, save_path]
	for path in path_list:
		if os.path.exists(path) is not True:
			os.makedirs(path)
	use_cuda = True
	model = torch.load(MODEL_PATH)
	del model.fc  # 去掉模型的全连接层，使得我们要可视化的那一层成为最后一层
	# print(model)
	grad_cam = GradCam(model, target_layer_names=["layer4"], use_cuda=use_cuda)
	image = []
	for s in os.listdir(img_path):
		image.append((cv2.imread(os.path.join(img_path, s), 1), s))

	for i, img_raw in enumerate(image):
		img, name = img_raw
		img_new = cv2.resize(img, (224, 224))
		# 对于图片进行预处理，使其变成能输入进网络里的tensor
		img = np.float32(cv2.resize(img, (224, 224))) / 255
		inputs = preprocess_image(img)
		inputs.required_grad = True

		# 选择一个分类模型的一个输出端口索引来可视化
		# 如果填None，则默认可视化模型的预测分类
		target_index = None

		# 获得grad_cam
		mask = grad_cam(inputs, target_index)

		# 融合grad_cam和原图并保存
		show_cam_on_image(img, mask, name, save_path)

		# 保存grad图
		# gb_model = GuidedBackpropReLUModel(model=torch.load(MODEL_PATH), use_cuda=use_cuda)
		# gb = gb_model(inputs, index=target_index)
		# utils.save_image(torch.from_numpy(gb), os.path.join(save_path, 'gb_{}.jpg'.format(i)))
		#
		# # 保存用cam对于grad图进行遮罩后的图
		# cam_mask = np.zeros(gb.shape)
		# for j in range(0, gb.shape[0]):
		# 	cam_mask[j, :, :] = mask
		# cam_gb = np.multiply(cam_mask, gb)
		# utils.save_image(torch.from_numpy(cam_gb), os.path.join(save_path, 'cam_gb_{}.jpg'.format(i)))
