# specify device
from torch import cuda
import torch.nn as nn
import transformers
from transformers import DistilBertTokenizer, DistilBertModel
import warnings
import torch
import torch
from transformers import DistilBertTokenizer, DistilBertModel
    
# Load the BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')  

def loadBERT(location, modelname):
    
    """
    Load the pretrained BERT model for text description data.
    """
    
    warnings.filterwarnings("ignore")

    device = 'cuda' if cuda.is_available() else 'cpu'
    
    bert = DistilBertModel.from_pretrained('distilbert-base-uncased')

    
    class BERT(nn.Module):
        def __init__(self, bert):

            super(BERT, self).__init__()

            # Distil Bert model
            self.bert = bert
            ## Additional layers
            # Dropout layer
            self.dropout = nn.Dropout(0.1)
            # Relu activation function
            self.relu =  nn.ReLU()
            # Dense layer 1
            self.fc1 = nn.Linear(768, 512)
            # Dense layer 2 (Output layer)
            self.fc2 = nn.Linear(512, 2)
            # Softmax activation function
            self.softmax = nn.LogSoftmax(dim=1)

        #define the forward pass
        def forward(self, **kwargs):

            #pass the inputs to the model BERT  
            cls_hs = self.bert(**kwargs)
            hidden_state = cls_hs.last_hidden_state
            pooler = hidden_state[:, 0]

            # dense layer 1        
            x = self.fc1(pooler)
            # ReLU activation
            x = self.relu(x)
            # Drop out
            x = self.dropout(x)
            # dense layer 2
            x = self.fc2(x)
            # apply softmax activation
            x = self.softmax(x)

            return x
        
    model = BERT(bert)
    # push the model to GPU
    model = model.to(device)

    # Load trained model (colab)
    try:
        model_save_name = modelname
        path = location + model_save_name
        model.load_state_dict(torch.load(path))
        print('Cuda Success')

    except:
        model_save_name = modelname
        path = location + model_save_name
        model.load_state_dict(torch.load(path, 
                                         map_location=torch.device('cpu')))
        print('CPU Success')

    model.eval()
    
    return model

def SpanPredictor(span, model, pred_values=False):

    """
    Uses a trained bert classifier to see if a span
    belongs to a species description or otherwise.
    """
        
    with torch.no_grad():
        # Tokenize input
        inputs = tokenizer(span, return_tensors="pt", truncation=True)
        # Predict class
        outputs = model(**inputs)
        # Get prediction values
        exps = torch.exp(outputs)
        # Get class
        span_class = exps.argmax(1).item()

        # Print the prediction values
        if pred_values:
            return span_class, exps[0]
        else:    
            return span_class
        