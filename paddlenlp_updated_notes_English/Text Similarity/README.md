## Description

There are many tasks related to text similarity and text matching classification task is just one of them. More about text matching can be found out in my new repository [text-matching-explained](https://github.com/jaaack-wang/text-matching-explained) (also including building and training models using Pytorch and TensorFlow).

More detailed explanations on the five or three (broadly) neural models are coming soon. 

The general architecture for the models constructed here can be seen below:

**Please note that**, 

- Another way to call the first embedding layer (for the text pair) is **representation layer**; and another way to call the second embedding layer is **matching layer**, whichever is easier for you to digest. 
- every arrow represents a process, or more specifically a function (let's be philosophical, no function at all is also a function, i.e., f(x)=x) that values from previous layer needs to go through; 
- the number of layers can be arbitrarily large, but our models are basically all four layers (including the input layer, but excluding the inserted CNN, RNNs, which are taken as a function here for illustration).
- There should be many ways to convert two embeddings into one aggregated embedding, but here we simply concatenate them together, meaning [1, 2, 3] + [4, 5, 6] = [1, 2, 3, 4, 5, 6]; [[1, 2, 3]] + [[4, 5, 6]] = [[1, 2, 3], [4, 5, 6]] or [[1, 2, 3, 4, 5, 6]] (dependending one axis you concatenate). Our models do the latter. 
- **The key difference among these models** how the initial two embeddings are processed or **encoded** before they are concatenated together. If you read the source codes for the five models, you can see the **encoders** for the five models what makes a model a model. 

<p align='center'>
 <img align="center" width='750' height='550' src="./imgs/one_text_pair_one_label_clf.png">
</p>
