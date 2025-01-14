import numpy as np
from sklearn.metrics import confusion_matrix


def cal_mice(TN,FP,FN,TP):
    N = TN+FP+FN+TP
    R1 = (TP+FN)/N
    R2 = (FP+TN)/N
    A = (TP+TN)/N
    
    MICE = (A-R1**2 - R2**2) / (1-R1**2 - R2**2)

    UA1 = TP/(TP+FP)
    UA2 = TN/(FN+TN)

    PA1 = TP/(TP+FN)
    PA2 = TN/(FP+TN)

    E_UA1 = (UA1 - R1) / (1 - R1)
    E_UA2 = (UA2 - R2) / (1 - R2)

    E_PA1 = (PA1 - R1) / (1 - R1)
    E_PA2 = (PA2 - R2) / (1 - R2)
    
    print('Overall Accuracy is:',A)
    print('MICE is:',MICE)
    print('User Accuracy 1 is:',UA1)
    print('User Accuracy 2 is:',UA2)
    print('Producer Accuracy 1 is:',PA1)
    print('Producer Accuracy 2 is:',PA2)
    print('CICE 1 is:',E_UA1)
    print('CICE 2 is::',E_UA2)
    print('OICE 1 is:',E_PA1)
    print('OICE 2 is::',E_PA2)
    return MICE

def metrics(gts, predictions):
    cm = confusion_matrix(
            y_true = gts,
            y_pred = predictions,
            labels=[0,1,2,3])
    
#     tn, fp, fn, tp = confusion_matrix(gts, predictions).ravel()
    print("Confusion matrix :")
    print(cm)
    
    print("---")
    
    # Compute global accuracy
    total = sum(sum(cm))
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)
    print("{} pixels processed".format(total))
    print("Total accuracy : {}%".format(accuracy))
    
    print("---")
    
    # Compute F1 score
    F1Score = np.zeros(len(label_values))
    for i in range(len(label_values)):
        try:
            F1Score[i] = 2. * cm[i,i] / (np.sum(cm[i,:]) + np.sum(cm[:,i]))
        except:
            # Ignore exception if there is no element in class i for test set
            pass
    print("F1Score :")
    for l_id, score in enumerate(F1Score):
        print("{}: {}".format(label_values[l_id], score))

    print("---")
        
    # Compute kappa coefficient
    total = np.sum(cm)
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / float(total*total)
    kappa = (pa - pe) / (1 - pe);
    print("Kappa: " + str(kappa))
#     cal_mice(tn,fp,fn,tp)
    return accuracy

def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb.cuda()).float().mean()
