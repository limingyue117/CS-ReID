from models.Classifier import Classifier, NormalizedClassifier
from models.ResNet import ResNet50
from models.ConvNeXt import convnext_tiny

def build_model(config, num_class):
	# Build backbone
	print("Initializing model: {}".format(config.MODEL.NAME))
	if config.MODEL.NAME == 'resnet50':
		model = ResNet50(res4_stride=config.MODEL.RES4_STRIDE,modelname=config.MODEL.POOLINGNAME, dim_feature=config.MODEL.FEATURE_DIM)
	elif config.MODEL.NAME == 'convnext':
		model = convnext_tiny(pretrained=True)
	else:
		raise KeyError("Invalid model: '{}'".format(config.MODEL.NAME))
	print("Model size: {:.5f}M".format(sum(p.numel() for p in model.parameters())/1000000.0))

	# Build classifier
	if config.LOSS.CLA_LOSS in ['crossentropy', 'crossentropylabelsmooth']:
		classifier = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_class)
	else:
		classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_class)
	
	if config.DATA.DATASET == 'prcc_ske':
		model1 = ResNet50(res4_stride=config.MODEL.RES4_STRIDE,modelname=config.MODEL.POOLINGNAME, dim_feature=config.MODEL.FEATURE_DIM)
		classifier1 = Classifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_class)
		classifier2 = Classifier(feature_dim=config.MODEL.FEATURE_DIM*2, num_classes=num_class)
		return model, model1, classifier, classifier1, classifier2
	return model, classifier  

def build_classifier(config, num_class):

	classifier = NormalizedClassifier(feature_dim=config.MODEL.FEATURE_DIM, num_classes=num_class)
	return classifier