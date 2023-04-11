from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_curve, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class EvaluateModel:
    def __init__(self, y, y_prob, thresh_search=True, best_thresh=0.5):
        num_classes = len(set(y))
        if num_classes == 2:
            self.binary_eval(y, y_prob, thresh_search, best_thresh)    
        else:
            self.multiclass_eval(y, y_prob)


    def check_imbalance(self, labels):
        labels = np.asarray(labels, dtype = 'int')
        class_counts = np.bincount(labels)
        class_proportions = class_counts / float(len(labels))
        max_proportion = np.max(class_proportions)
        min_proportion = np.min(class_proportions)
        imbalance_ratio = max_proportion / min_proportion

        return imbalance_ratio


    def binary_eval(self, y, y_prob, thresh_search, best_thresh):
        scores_dict={}
        scores_dict['aupr'] = average_precision_score(y, y_prob)
        scores_dict['auroc'] = roc_auc_score(y, y_prob)

        check_imbalance = self.check_imbalance(y)
        
        if check_imbalance >= 2:
            scores_dict['aupr_pr'], scores_dict['aupr_rec'], scores_dict['aupr_thresholds'] = precision_recall_curve(y, y_prob)
            _best_thresh, scores_dict['aupr_best_pr'], scores_dict['aupr_best_rec'] = self.get_threshold(precision=scores_dict['aupr_pr'], recall=scores_dict['aupr_rec'], thresholds=scores_dict['aupr_thresholds'])
        else:
            scores_dict['fpr'], scores_dict['tpr'], scores_dict['roc_thresh'] = roc_curve(y_true=y, y_score=y_prob)
            _best_thresh, scores_dict['roc_best_fpr'], scores_dict['roc_best_tpr'] = self.get_threshold(precision=scores_dict['fpr'], recall=scores_dict['tpr'], thresholds=scores_dict['roc_thresh'])

        if thresh_search:
            best_thresh = _best_thresh

        y_pred = (y_prob >= best_thresh).astype(int)
        scores_dict['y_pred'] = y_pred
        scores_dict['y_prob'] = y_prob
        scores_dict['y'] = y
  
        scores_dict['acc'] = accuracy_score(y, y_pred)
        scores_dict['F1'] = f1_score(y, y_pred)
        scores_dict['precision'] = precision_score(y, y_pred)
        scores_dict['recall'] = recall_score(y, y_pred)
        
        scores_dict['cnf_matrix'] = confusion_matrix(y_true=y, y_pred=y_pred)
        tn, fp, fn, tp = scores_dict['cnf_matrix'].ravel()
        
        scores_dict['specificity'] = tn / (tn + fp)
        scores_dict['best_thresh'] = best_thresh
        scores_dict['_best_thresh'] = _best_thresh
        scores_dict['classification_report']=classification_report(y, y_pred, zero_division=0)

        self.scores = scores_dict


    def multiclass_eval(self, y, y_prob):
        scores_dict={}
        y_pred = np.argmax(y_prob, axis=1)
        y_prob_ = np.max(y_prob, axis=1)
        
        scores_dict['y_pred'] = y_pred
        scores_dict['y_prob'] = y_prob_
        scores_dict['y'] = y
        scores_dict['acc'] = accuracy_score(y, y_pred)
        scores_dict['precision'] = precision_score(y, y_pred, average='weighted')
        scores_dict['recall'] = recall_score(y, y_pred, average='weighted')
        scores_dict['F1'] = f1_score(y, y_pred, average='weighted')
        scores_dict['cnf_matrix'] = confusion_matrix(y, y_pred)
        scores_dict['classification_report']=classification_report(y, y_pred, zero_division=0)

        self.scores = scores_dict


    def get_threshold(self, vector1, vector2, thresholds):
        thresh_filt = []
        fscore = []
        n_thresh = len(thresholds)
        for idx in range(n_thresh):
            if vector1[idx] == 0 and vector2[idx] == 0:
                curr_f1 = 0.
            else:
                curr_f1 = (2 * vector1[idx] * vector2[idx]) / \
                    (vector1[idx] + vector2[idx])
            if not (np.isnan(curr_f1)):
                fscore.append(curr_f1)
                thresh_filt.append(thresholds[idx])
        ix = np.argmax(np.array(fscore))
        best_thresh = thresh_filt[ix]
        return best_thresh, vector1[ix], vector2[ix]


    def plot_confusion_matrix(self, path, model_name, target_names):
        cm = self.cnf_matrix
        ax = sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')

        ax.set_title(model_name+'\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values')

        ax.xaxis.set_ticklabels(target_names)
        ax.yaxis.set_ticklabels(target_names)

        plt.savefig(path+'_cm.png',bbox_inches = "tight")
        plt.close()