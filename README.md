# sparkastML

This repository houses the machine learning components for the [sparkast](https://github.com/alikia2x/sparkast) project.

The primary objective of this project is to enhance the search functionality of sparkast, allowing users to receive real-time answers as they type their queries.

## Intention Classification

The model located in the `/intention-classify` directory is designed to categorize user queries into predefined classes.

We utilize a Convolutional Neural Network (CNN) architecture in conjunction with an Energy-based Model for open-set recognition.

This model is optimized to be lightweight, ensuring it can run on a wide range of devices, including within the browser environment.

For a detailed explanation of how it works, you can refer to [this blog post](https://blog.alikia2x.com/en/posts/sparkastml-intention/).

## Dataset

To support the development of Libre Intelligence, we have made a series of datasets publicly available. You can access them [here](./dataset/public/README.md).
