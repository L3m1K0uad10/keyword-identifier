# Programming Language Keyword Identifier

This project explores supervised learning on text data by building a model that can identify whether a given word(token) is a programming language keyword (or reserved word) and, if so, list all the programming languages in which it is used. The project leverages a custom CSV dataset that contains both reserved keywords and common non-keyword identifiers.

## Project Overview

The project is structured around two primary tasks:

1. **Binary Classification:**  
   Determine whether a given word is a programming language keyword (reserved word) or not.

2. **Multi-label Classification (Optional/Secondary):**  
   For words identified as keywords, output a list of programming languages that reserve that keyword.

While the core task is framed as a binary classification problem, the project incorporates significant **Natural Language Processing (NLP)** elements. Since the input data consists of text tokens, various NLP techniques are applied to improve feature extraction and generalization—especially for tokens that may not have been seen during training.

## NLP Considerations

- **Feature Extraction:**  
  NLP techniques enable the model to extract and represent meaningful features from each token. For example, reserved keywords typically follow specific patterns (e.g., short, all lowercase) compared to user-defined identifiers (e.g., `commentParser` in camelCase).  
  - *Character-Level Features:* Using n-grams or subword tokenization helps capture these patterns.
  - *Normalization:* Converting tokens to a consistent case or format can reduce variability.

- **Generalization:**  
  By applying NLP-based processing, the model can better handle out-of-vocabulary tokens and generalize its predictions. This means that even if a token like `"commentParser"` was not present in the training data, the model can use its learned representations to correctly classify it as a non-keyword.

## Dataset

The CSV dataset (`keywords.csv`) includes three columns:
- **token:** The token to be evaluated.
- **is_keyword:** A binary indicator (`1` if the token is a reserved keyword in at least one language, `0` if not).
- **languages:** A comma-separated list of programming languages where the token is reserved. For non-keyword tokens, this field is set to `"None"`.

The dataset encompasses keywords from popular programming languages such as **Python, C, C++, Java, JavaScript, and C#**. It also includes additional tokens that, while common in programming, are not reserved keywords.

### Sample Data (shuffled)
```csv
token,is_keyword,languages
temp,0,"None"
@Override,1,"Java"
process,0,"None"
abstract,1,"Java, C#"
operator,1,"C++, C#"
constructor,0,"None"
False,1,"Python"
auto,1,"C"
...
