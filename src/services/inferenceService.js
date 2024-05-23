const tf = require('@tensorflow/tfjs-node');
const InputError = require('../exceptions/InputError');

async function predictClassification(model, image) {
  try {
    const tensor = tf.node
      .decodeJpeg(image)
      .resizeNearestNeighbor([224, 224])
      .expandDims()
      .toFloat()

    const prediction = model.predict(tensor);
    const score = await prediction.data();
    const confidenceScore = Math.max(...score) * 100;

    const classes = ['Melanocytic nevus', 'Squamous cell carcinoma', 'Vascular lesion'];

    const classResult = tf.argMax(prediction, 1).dataSync()[0];
    const label = classes[classResult];

    let suggestion, resultLabel;

    if (label === 'Melanocytic nevus') {
      resultLabel = "Non-Cancer";
      suggestion = "Anda Sehat!";
    }
  
    if (label === 'Squamous cell carcinoma') {
      resultLabel = "Cancer";
      suggestion = "Segera Periksa ke Dokter!";
    }
  
    if (label === 'Vascular lesion') {
      resultLabel = "Non-Cancer";
      suggestion = "Anda Sehat!";
    }

    return { confidenceScore, resultLabel, suggestion };
  } catch (error) {
    throw new InputError(`Terjadi kesalahan input: ${error.message}`);
  }
}

module.exports = predictClassification;
