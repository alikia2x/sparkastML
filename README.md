# sparkastML

This repository contains the machine learning components for the [sparkast](https://github.com/alikia2x/sparkast) project.

The main goal of this project is to improve the search functionality of sparkast, enabling users to receive real-time answers as they type their queries.

## Intention Classification

The model in the `/intention-classify` directory is designed to categorize user queries into predefined classes.

We use a Convolutional Neural Network (CNN) architecture combined with an Energy-based Model for open-set recognition.

This model is optimized to be lightweight, ensuring it can run on a wide range of devices, including within the browser environment.

For a detailed explanation of how it works, refer to [this blog post](https://blog.alikia2x.com/en/posts/sparkastml-intention/).

## Translation

Language barriers are one of the biggest obstacles to communication between civilizations. In modern times, with the development of computer science and artificial intelligence, machine translation is bridging this gap and building a modern Tower of Babel.

Unfortunately, many machine translation systems are owned by commercial companies, which seriously hinders the development of freedom and innovation.

Therefore, sparkastML is on a mission to challenge commercial machine translation. We decided to tackle the translation between Chinese and English first. These are two languages with a long history and a large number of users. Their writing methods and expression habits are very different, which brings challenges to the project.

For more details, visit [this page](./translate/README.md).

## Dataset

To support the development of Libre Intelligence, we have made a series of datasets publicly available. You can access them [here](./dataset/public/README.md).
