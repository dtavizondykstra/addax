# addax

Text analysis library

## Installation

```bash
$ pip install addax
```

## Usage

`addax` can be used to clean target text and then apply sentiment analysis to it in order to return a sentiment score and label, as follows:

```
from addax.addax import (
    read_csv,
    process_sentiment,
)

df = read_csv("file.csv") # path to your file
process_sentiment(df=df, target_column="comment", include_subjectivity=True, label=True)

```

## Contributing

Interested in contributing? Check out the contributing guidelines. Please note that this project is released with a Code of Conduct. By contributing to this project, you agree to abide by its terms.

## License

`addax` was created by Dalia Tavizon-Dykstra. It is licensed under the terms of the MIT license.

## Credits

`addax` was created with [`cookiecutter`](https://cookiecutter.readthedocs.io/en/latest/) and the `py-pkgs-cookiecutter` [template](https://github.com/py-pkgs/py-pkgs-cookiecutter).
