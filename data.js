export class YoloImageData {
  constructor() {
  }

  async load() {
  }

  nextTrainBatch(batchSize) {
  }

  nextTestBatch(batchSize) {
  }

  nextBatch(batchSize, data, index) {
    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return {xs, labels};
  }
}
