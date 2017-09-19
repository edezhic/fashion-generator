const path = require('path');

module.exports = {
  entry: './generator.js',
  output: {
    path: path.resolve(__dirname, 'dist'),
    filename: 'bundle.js'
  }
};
