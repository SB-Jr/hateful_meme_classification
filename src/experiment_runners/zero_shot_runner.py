from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from src.utils.hmc_dataset import HatefulMemesDataset
from tqdm.notebook import tqdm


def get_zero_shot_model_objects(dataset_path, model_name, device_map='cuda'):
    model_processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForZeroShotImageClassification.from_pretrained(model_name, device_map=device_map)
    def process_img(img):
        return model_processor(
            images=img,
            text=['This is meme is not hateful', 'This meme is hateful'], 
            return_tensors='pt',
            padding=True
        )
    processed_ds = HatefulMemesDataset(dataset_path, limit_to_examples=100, transform=process_img)
    return processed_ds, model


def get_model_metrics(dataset, model):
    
    true_labels = []
    predicted_labels = []

    for data in tqdm(dataset, total=len(dataset)):
        model_input = data[0].to(model.device)
        label = data[1]
        output = model(**model_input).logits_per_image.softmax(dim=1)[0]
        pred = output.cpu().argmax().numpy()
        true_labels.append(label)
        predicted_labels.append(pred)
    
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

    final_outcome = {
        'accuracy': accuracy_score(true_labels, predicted_labels),
        'precision': precision_score(true_labels, predicted_labels),
        'recall': recall_score(true_labels, predicted_labels),
        'f1_score': f1_score(true_labels, predicted_labels),
        'true_pos': tp,
        'true_neg': tn,
        'false_pos': fp,
        'false_neg': fn
    }
    return final_outcome
    