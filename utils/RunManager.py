
import time
import pandas as pd
from collections import OrderedDict
import datetime as datetime
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix, MulticlassPrecision, MulticlassRecall, MulticlassAUPRC, MulticlassAUROC, MulticlassPrecisionRecallCurve
#import wandb
import os

class RunManager():
    def __init__(self, number_of_classes):
        self.epoch_count = 0
        self.epoch_loss = 0
        self.epoch_test_loss = 0
        self.epoch_num_correct = 0
        self.epoch_test_num_correct = 0
        self.epoch_start_time = None
        self.epoch_accuracy = 0
        self.epoch_test_accuracy = 0
        self.epoch_precision = 0
        self.epoch_recall = 0
        self.epoch_F1 = 0
        self.epoch_weighted_F1 = 0
        self.epoch_auprc = 0
        self.epoch_auroc = 0
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        self.model = None
        self.loader = None
        self.run_name = None
        self.best_val_loss = 1_000_000.
        self.kfold_validation_loss = 0
        self.epoch_confusion = []

        self.metric_precision = MulticlassPrecision(num_classes=number_of_classes)
        self.metric_recall = MulticlassRecall(num_classes=number_of_classes)
        self.metric_accuracy = MulticlassAccuracy(average='micro', num_classes=number_of_classes)
        self.metric_F1_weighted = MulticlassF1Score(average='weighted', num_classes=number_of_classes)
        self.metric_F1 = MulticlassF1Score(average='macro', num_classes=number_of_classes)
        self.metric_confusion = MulticlassConfusionMatrix(num_classes=number_of_classes)
        self.metric_AUPRC = MulticlassAUPRC(num_classes=number_of_classes)
        self.metric_AUROC = MulticlassAUROC(num_classes=number_of_classes)
        self.metric_confusion = MulticlassConfusionMatrix(num_classes=number_of_classes)
        
    
    def begin_run(self, run, model, loader, device, run_name):
        self.run_start_time = time.time()
        self.epoch_count = 0
        self.run_params = run
        self.run_count += 1
        self.model = model
        self.loader = loader
        self.run_name = run_name

    def end_run(self):
        self.epoch_count = 0
        

    def begin_epoch(self):
        self.epoch_start_time = time.time()
        self.epoch_count+=1
        self.epoch_loss = 0
        self.epoch_test_loss = 0
        self.epoch_num_correct = 0
        self.epoch_test_num_correct = 0
        self.epoch_accuracy = 0
        self.epoch_test_accuracy = 0
        self.epoch_precision = 0
        self.epoch_recall = 0
        self.epoch_F1 = 0
        self.epoch_weighted_F1 = 0
        self.epoch_auprc = 0
        self.epoch_auroc = 0

        
    def calculate_metrics(self):
       
        self.epoch_precision = self.metric_precision.compute().item()
        self.epoch_recall = self.metric_recall.compute().item()
        self.epoch_F1 = self.metric_F1.compute().item()
        self.epoch_weighted_F1 = self.metric_F1_weighted.compute().item()
        self.epoch_auprc = self.metric_AUPRC.compute().item()
        self.epoch_auroc = self.metric_AUROC.compute().item()
        self.epoch_auroc = self.metric_AUROC.compute().item()
        self.epoch_confusion = self.metric_confusion.compute()


    def end_epoch(self):

        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        results = OrderedDict()
        results["run"] = self.run_count
        results["epoch"] = self.epoch_count
        results["loss"] = self.epoch_loss
        results["accuracy"] = self.epoch_accuracy
        results["test loss"] = self.epoch_test_loss
        results["test accuracy"] = self.epoch_test_accuracy
        results["precision"] = self.epoch_precision
        results['recall'] = self.epoch_recall
        results["F1 Score"] = self.epoch_F1
        results["Weighted F1 Score"] = self.epoch_weighted_F1
        results["auprc"] = self.epoch_auprc
        results["auroc"] = self.epoch_auroc
        results["epoch duration"] = epoch_duration
        results["run duration"] = run_duration

        for k,v in self.run_params._asdict().items(): 
            results[k] = v
        
        self.run_data.append(results)
        df = pd.DataFrame.from_dict(self.run_data, orient='columns')


        confusion_matrix = self.metric_confusion.compute()    
        
        cm_numpy = confusion_matrix.numpy()
        cm_df = pd.DataFrame(cm_numpy)

        if not os.path.exists(f'results/confusion_matrix/{self.run_name}'):
            # Create the directory
            os.makedirs(f'results/confusion_matrix/{self.run_name}')


        cm_df.to_csv(f'results/confusion_matrix/{self.run_name}/{self.run_name}_{self.epoch_count}.csv')

        #self.cm_artifact.add_file(local_path=f'results/confusion_matrix/{self.run_name}/{self.run_name}_{self.epoch_count}.csv')
        #wandb.log_artifact(self.cm_artifact) 

        self.metric_accuracy.reset()
        self.metric_precision.reset()
        self.metric_recall.reset()
        self.metric_F1.reset()
        self.metric_F1_weighted.reset()
        self.metric_confusion.reset()
        self.metric_AUPRC.reset()
        self.metric_AUROC.reset()
        self.metric_confusion.reset()

    def save(self, run_name):
        pd.DataFrame.from_dict(
            self.run_data,
            orient='columns'
        ).to_csv(f'results/{run_name}.csv')

        # with open(f'results/{fileName}.json', 'w', encoding='utf-8') as f:
        #     json.dump(self.run_data, f, ensure_ascii=False, indent=4)


    def update_metrics(self,test_preds, test_labels):
        self.metric_precision.update(test_preds.argmax(dim=1), test_labels)
        self.metric_recall.update(test_preds.argmax(dim=1), test_labels)
        self.metric_accuracy.update(test_preds.argmax(dim=1), test_labels)
        self.metric_F1.update(test_preds.argmax(dim=1), test_labels)
        self.metric_F1_weighted.update(test_preds.argmax(dim=1), test_labels)
        self.metric_confusion.update(test_preds.argmax(dim=1), test_labels)
        self.metric_AUPRC.update(test_preds, test_labels)
        self.metric_AUROC.update(test_preds, test_labels)
        self.metric_confusion.update(test_preds, test_labels)


    
    # def log_test_predictions(dataset, images, labels, outputs, predicted, image_table, log_counter, images_to_log): 
    #     scores = F.softmax(outputs.data, dim=1)
    #     log_scores = scores.cpu().numpy()
    #     log_images = images.cpu().numpy()
    #     log_labels = labels.cpu().numpy()
    #     log_preds = predicted.cpu().numpy()

    #     _id = 0

    #     for image, label, pred, score in zip(log_images, log_labels, log_preds, log_scores):
    #         img_id = str(_id) + "_" + str(log_counter)
    #         if label != pred:
    #             image = image.transpose(1, 2, 0)
    #             image_table.add_data(img_id, wandb.Image(image), dataset.classes[pred], dataset.classes[label], *score)
    #             _id += 1
    #         if _id == images_to_log:
    #             break