import pickle
from collections import Counter
import random
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score


# Function to perform majority voting on three lists
def majority(list1, list2, list3):
    majority_list = []
    for a, b, c in zip(list1, list2, list3):
        vote_count = Counter([a, b, c])
        if vote_count.most_common(1)[0][1] == 1: #All predictions are different
            majority_element = random.randint(0, 2)
        majority_element = vote_count.most_common(1)[0][0] #label with the most popularity
        majority_list.append(majority_element)
    return majority_list

# Load the lists from pickle files
with open('./bert_predictions.pkl', 'rb') as f:
    bert = pickle.load(f)

with open('./roberta_predictions.pkl', 'rb') as f:
    roberta = pickle.load(f)

with open('./xlnet_predictions.pkl', 'rb') as f:
    xlnet = pickle.load(f)
    
with open('./true_labels.pkl', 'rb') as f:
    true_labels = pickle.load(f)

# Perform majority voting
final_prediction = majority(bert, roberta, xlnet)

# Print the resulting list
print("Predictions after majority voting:", final_prediction)

def calculate_metrices(true_labels, predicted_labels, model_name, append_stats = True):
    precision = precision_score(true_labels, predicted_labels, average='macro')
    recall = recall_score(true_labels, predicted_labels, average='macro')
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='macro')
    if append_stats:
        f = open('./output_metrics.txt', 'a')
    else:
        f = open('./output_metrics.txt', 'w')
    f.write(f"---------------Results from Model {model_name}---------------\n\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write("\n")
    
    f.close()
    
    

calculate_metrices(true_labels, bert, "Bert", False)
calculate_metrices(true_labels, roberta, "Roberta")
calculate_metrices(true_labels, xlnet, "XLNet")
calculate_metrices(true_labels, final_prediction, "Encell")

print("Metrices successfully calculated. Please check output_metrices.txt file") 