//import * as tf from '@tensorflow/tfjs';

const IMAGE_SIZE = 416 * 416 * 3;
const NUM_CLASSES = 20;
//const NUM_DATASET_ELEMENTS = 5011;
const NUM_DATASET_ELEMENTS = 100;

const TRAIN_TEST_RATIO = 4 / 5;

const NUM_TRAIN_ELEMENTS = Math.floor(TRAIN_TEST_RATIO * NUM_DATASET_ELEMENTS);
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

const MNIST_IMAGES_SPRITE_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_images.png';
const MNIST_LABELS_PATH = 'https://storage.googleapis.com/learnjs-data/model-builder/mnist_labels_uint8';

const XML_ANNOTATION_PATH = 'http://localhost:8080/annotations.json';

const CLASSES = {
  aeroplane: 0,
  bicycle: 0,
  bird: 0,
  boat: 0,
  bottle: 0,
  bus: 0,
  car: 0,
  cat: 0,
  chair: 0,
  cow: 0,
  diningtable: 0,
  dog: 0,
  horse: 0,
  motorbike: 0,
  person: 0,
  pottedplant: 0,
  sheep: 0,
  sofa: 0,
  train: 0,
  tvmonitor: 0,
};

export class YoloImageData {
  constructor() {
    this.shuffledTrainIndex = 0;
    this.shuffledTestIndex = 0;

    this.annotations = [];
  }

  async load() {
    let filenames = [];
    await fetch(XML_ANNOTATION_PATH)
      .then(response => {
        return response.json().then(json => {
          for (var value of json.names) {
            filenames.push(value);
          }
        });
      });

    for (var value of filenames) {
      let url = `./VOCdevkit/VOC2007/Annotations/${value}`;
      let xhr = new XMLHttpRequest ();
      xhr.addEventListener('load', () => {
        let xml = xhr.responseXML.getElementsByTagName('annotation').item(0);
        let objectElems = xml.getElementsByTagName('object');
        let objects = [];
        for (var elm of objectElems) {
          objects.push ({
            'name': elm.getElementsByTagName('name').item(0).textContent,
            'truncated': elm.getElementsByTagName('truncated').item(0).textContent,
            'difficult': elm.getElementsByTagName('difficult').item(0).textContent,
            'bbox': [
              elm.getElementsByTagName('xmin').item(0).textContent,
              elm.getElementsByTagName('ymin').item(0).textContent,
              elm.getElementsByTagName('xmax').item(0).textContent,
              elm.getElementsByTagName('ymax').item(0).textContent
            ]
          });
        }
        let annotation = {
          'filename': xml.getElementsByTagName('filename').item(0).textContent,
          'width': xml.getElementsByTagName('width').item(0).textContent,
          'height': xml.getElementsByTagName('height').item(0).textContent,
          'depth': xml.getElementsByTagName('depth').item(0).textContent,
          'segment': xml.getElementsByTagName('segmented').item(0).textContent,
          'objects': objects
        }
        
        this.annotations.push(annotation);
      }, false);
      xhr.open('GET', url, false);
      xhr.send();
    }

    // Create shuffled indices into the train/test set for when we select a
    // random dataset element for training / validation.
    this.trainIndices = tf.util.createShuffledIndices(NUM_TRAIN_ELEMENTS);
    this.testIndices = tf.util.createShuffledIndices(NUM_TEST_ELEMENTS);

    // Slice the the images and labels into train and test sets.
    // this.trainImages = this.datasetImages.slice(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    // this.testImages = this.datasetImages.slice(IMAGE_SIZE * NUM_TRAIN_ELEMENTS);
    // this.trainLabels = this.datasetLabels.slice(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
    // this.testLabels = this.datasetLabels.slice(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  }

  nextTrainBatch(batchSize) {
    return this.nextBatch(
      batchSize, [this.trainImages, this.trainLabels], () => {
        this.shuffledTrainIndex = (this.shuffledTrainIndex + 1) % this.trainIndices.length;
        return this.trainIndices[this.shuffledTrainIndex];
      });
  }

  nextTestBatch(batchSize) {
    return this.nextBatch(
      batchSize, [this.testImages, this.testLabels], () => {
        this.shuffledTestIndex = (this.shuffledTestIndex + 1) % this.testIndices.length;
        return this.testIndices[this.shuffledTestIndex];
    });
  }

  async nextBatch(batchSize, data, index) {
    const img = new Image();
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    const div = document.getElementById('image');
    div.appendChild(canvas);

    const batchImagesArray = new Float32Array(batchSize * IMAGE_SIZE);
    const batchLabelsArray = new Uint8Array(batchSize * NUM_CLASSES);
    const promises = [];

    for (let i = 0; i < batchSize; i++) {
      const idx = index();

      let annotation = this.annotations[idx];
      let url = `./VOCdevkit/VOC2007/JPEGImages/${annotation.filename}`;

      let imageSource = new Promise((resolve, reject) => { 
        let image = new Image();
        image.crossOrigin = "anonymous";
        image.src = url; 
        image.onload = function() {
//          canvas.width = image.naturalWidth; 
//          canvas.height = image.naturalHeight;
          canvas.width = 416;
          canvas.height = 416;
          ctx.drawImage(image, 0, 0);

          const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
          batchImagesArray.set(imageData, i * IMAGE_SIZE);

          const label = new Uint8Array((annotation) => {
            let classes = Object.assign({}, CLASSES);
            for (var obj of annotation.objects) {
              classes[obj.name]++;
            }
            return Object.values(classes);
          });

          batchLabelsArray.set(label, i * NUM_CLASSES);

          resolve();
        }
      });

      promises.push(imageSource);
    }

    await Promise.all(promises);
    
    const xs = tf.tensor2d(batchImagesArray, [batchSize, IMAGE_SIZE]);
    const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);

    return {xs, labels};
  }
}
