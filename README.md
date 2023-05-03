# N-Gram_TOFL_Evaluator
Uses a simple N-Gram language model to determine similarity of test TOEFLs to training TOEFLs.

# Use:
Run the program with the first argument as the filepath of the training data directory, and the second argument as the filepath to the target data directory (must all be in .txt format). This will report the perplexity of the target data relative to the training data using a simple trigram approach. Some sample data has been uploaded for training, and a framework provided in the code for automatic batch classification. 
