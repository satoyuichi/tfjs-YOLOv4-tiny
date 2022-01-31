// const tf = require('@tensorflow/tfjs-node');
// const tfvis = require('@tensorflow/tfjs-vis');

// const model = createModel();
// model.summary();
// model.save("file://mymodel");
// model.compile({
//   loss: 'categoricalCrossentropy',
//   optimizer: 'adam'
// });
// await train(model, data);

export class Yolov4Model {
  createModel() {
    const input = tf.input({shape: [416, 416, 3]});

    let conv2d_01 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: 32,
      batchSize: 1,
      strides: 2,
      padding: 'same',
      dilationRate: [1, 1],
      name: 'conv2d_01'
    }).apply(input);
    let leakyReLU_01 = tf.layers.leakyReLU({
      name: 'leakyReLU_01'
    }).apply(conv2d_01);

    let conv2d_02 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: 64,
      batchSize: 1,
      strides: 2,
      padding: 'same',
      dilationRate: [1, 1],
      name: 'conv2d_02'
    }).apply(leakyReLU_01);
    let leakyReLU_02 = tf.layers.leakyReLU({
      name: 'leakyReLU_02'
    }).apply(conv2d_02);

    let conv2d_03 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: 64,
      batchSize: 1,
      strides: 1,
      padding: 'same',
      dilationRate: [1, 1],
      name: 'conv2d_03'
    }).apply(leakyReLU_02);
    let leakyReLU_03 = tf.layers.leakyReLU({
      name: 'leakyReLU_03'
    }).apply(conv2d_03);

    let sublayers_01 = this.createSubLayers(4, leakyReLU_03, 32);
    let concat_01 = tf.layers.concatenate({name: `concat_01`}).apply([leakyReLU_03, sublayers_01]);

    let maxPool2d_01 = tf.layers.maxPooling2d({
      poolSize: [2, 2],
    }).apply(concat_01);

    let conv2d_07 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: 128,
      batchSize: 1,
      strides: 1,
      padding: 'same',
      dilationRate: [1, 1],
      name: 'conv2d_07'
    }).apply(maxPool2d_01);
    let leakyReLU_07 = tf.layers.leakyReLU({
      name: 'leakyReLU_07'
    }).apply(conv2d_07);

    let sublayers_02 = this.createSubLayers(7, leakyReLU_07, 64);
    let concat_02 = tf.layers.concatenate({name: `concat_02`}).apply([leakyReLU_07, sublayers_02]);

    let maxPool2d_02 = tf.layers.maxPooling2d({
      poolSize: [2, 2],
    }).apply(concat_02);

    let conv2d_08 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: 256,
      batchSize: 1,
      strides: 1,
      padding: 'same',
      dilationRate: [1, 1],
      name: 'conv2d_08'
    }).apply(maxPool2d_02);
    let leakyReLU_08 = tf.layers.leakyReLU({
      name: 'leakyReLU_08'
    }).apply(conv2d_08);

    let sublayers_03 = this.createSubLayers(10, leakyReLU_08, 128);
    let concat_03 = tf.layers.concatenate({name: `concat_03`}).apply([leakyReLU_08, sublayers_03]);

    let maxPool2d_03 = tf.layers.maxPooling2d({
      poolSize: [2, 2],
    }).apply(concat_03);

    let conv2d_09 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: 256,
      batchSize: 1,
      strides: 1,
      padding: 'same',
      dilationRate: [1, 1],
      name: 'conv2d_09'
    }).apply(maxPool2d_03);
    let leakyReLU_09 = tf.layers.leakyReLU({
      name: 'leakyReLU_09'
    }).apply(conv2d_09);

    let conv2d_10 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: 256,
      batchSize: 1,
      strides: 1,
      padding: 'same',
      dilationRate: [1, 1],
      name: 'conv2d_10'
    }).apply(leakyReLU_09);
    let leakyReLU_10 = tf.layers.leakyReLU({
      name: 'leakyReLU_10'
    }).apply(conv2d_10);

    //--------------------------------------------------------------------------------

    let conv2d_11 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: 128,
      batchSize: 1,
      strides: 1,
      padding: 'same',
      dilationRate: [1, 1],
      name: 'conv2d_11'
    }).apply(leakyReLU_10);
    let leakyReLU_11 = tf.layers.leakyReLU({
      name: 'leakyReLU_11'
    }).apply(conv2d_11);
    let upSample2d_01 = tf.layers.upSampling2d({
      name: `upSample2d_01`
    }).apply(leakyReLU_11);

    let concat_04 = tf.layers.concatenate({
      name: `concat_04`
    }).apply([sublayers_03, upSample2d_01])

    let conv2d_12 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: 256,
      batchSize: 1,
      strides: 1,
      padding: 'same',
      dilationRate: [1, 1],
      name: 'conv2d_12'
    }).apply(concat_04);
    let leakyReLU_12 = tf.layers.leakyReLU({
      name: 'leakyReLU_12'
    }).apply(conv2d_12);
    let conv2d_13 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: 75,
      batchSize: 1,
      strides: 1,
      padding: 'same',
      dilationRate: [1, 1],
      name: 'conv2d_13'
    }).apply(leakyReLU_12);

    //--------------------------------------------------------------------------------

    let conv2d_14 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: 512,
      batchSize: 1,
      strides: 1,
      padding: 'same',
      dilationRate: [1, 1],
      name: 'conv2d_14'
    }).apply(leakyReLU_10);
    let leakyReLU_14 = tf.layers.leakyReLU({
      name: 'leakyReLU_14'
    }).apply(conv2d_14);
    let conv2d_15 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: 75,
      batchSize: 1,
      strides: 1,
      padding: 'same',
      dilationRate: [1, 1],
      name: 'conv2d_15'
    }).apply(leakyReLU_14);

    //================================================================================

    let model = tf.model({
      inputs: input,
      outputs: [conv2d_15, conv2d_13]
    });

    return model;
  }

  createSubLayers(index, input, filters) {
    let sub_conv2d_01 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: filters,
      batchSize: 1,
      strides: 1,
      padding: 'same',
      dilationRate: [1, 1],
      name: `sub_conv2d_${index}`
    }).apply(input);
    let sub_leakyReLU_01 = tf.layers.leakyReLU({
      name: `sub_leakyReLU_${index++}`
    }).apply(sub_conv2d_01);

    let sub_conv2d_02 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: filters,
      batchSize: 1,
      strides: 1,
      padding: 'same',
      dilationRate: [1, 1],
      name: `sub_conv2d_${index}`
    }).apply(sub_leakyReLU_01);
    let sub_leakyReLU_02 = tf.layers.leakyReLU({
      name: `sub_leakyReLU_${index}`
    }).apply(sub_conv2d_02);

    let sub_concat_01 = tf.layers.concatenate({name: `sub_concat_${index++}`}).apply([sub_leakyReLU_01, sub_leakyReLU_02]);

    let sub_conv2d_03 = tf.layers.conv2d({
      kernelSize: [3, 3],
      filters: filters * 2,
      batchSize: 1,
      strides: 1,
      padding: 'same',
      dilationRate: [1, 1],
      name: `sub_conv2d_${index}`
    }).apply(sub_concat_01);
    let sub_leakyReLU_03 = tf.layers.leakyReLU({
      name: `sub_leakyReLU_${index}`
    }).apply(sub_conv2d_03);

    return sub_leakyReLU_03;
  }
}
