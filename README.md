# POS-classification-with-RNN

# Summary
	The aim of this project is to make a POS Tagger using RNN and a pretrained ELMo model. 
# Files
Below you can find files and description of this files
1) pos_assign4_RNN_ver_3.py code for RNN model.
2) pos_assign4_elmo_v1.py code for pretrained ELMo model
# Issues
	Loading RNN is included in the code of the RNN python file and is commented out. 
  Due to the custom metric and loss functions used loading the files requires the specifics of these custom functions
	The same applies for ELMo model but there is a problem loading a model trained and saved with ELMo. 
  The code on the python file loads the model but it does not do that properly. The model is included for reference reasons.
