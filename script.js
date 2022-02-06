import {YoloImageData} from './data.js';
import {Yolov4Model} from './model.js';

function init (){
  const surface = tfvis.visor().surface({ name: 'Input Data Examples', tab: 'Input Data'});
}

async function train(model, data) {
  const metrics = ['loss', 'val_loss', 'acc', 'val_acc'];
  const container = {
    name: 'Model Training', tab: 'Model', styles: { height: '1000px' }
  };
  const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

  const BATCH_SIZE = 10;
//  const TRAIN_DATA_SIZE = 100;
  const TRAIN_DATA_SIZE = 10;
  const TEST_DATA_SIZE = 20;
  
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
    return [];
    // return [
    //   d.xs.reshape([TRAIN_DATA_SIZE, 416, 416, 3]),
    //   d.labels
    // ];
  });

  // const [testXs, testYs] = tf.tidy(() => {
  //   const d = data.nextTestBatch(TEST_DATA_SIZE);
  //   return [
  //     d.xs.reshape([TEST_DATA_SIZE, 416, 416, 3]),
  //     d.labels
  //   ];
  // });

 // return model.fit(trainXs, trainYs, {
 //    batchSize: BATCH_SIZE,
 //    validationData: [testXs, testYs],
 //    epochs: 10,
 //    shuffle: true,
 //    callbacks: fitCallbacks
 //  });  
}

async function run() {
  const data = new YoloImageData();
  const yolov4 = new Yolov4Model();
  const model = yolov4.createModel();

  init();
  
  tfvis.show.modelSummary({name: 'Model Architecture', tab: 'Model'}, model);

  await data.load();
  await train(model, data);
}

document.addEventListener('DOMContentLoaded', run);
