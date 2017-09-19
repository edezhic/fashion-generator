# Fashion Generator
Live version of this repo - [Fashion GAN Online](http://cognitivechaos.com/playground/fashion-gan/)

GAN model trained in Tensorflow and then ported to browsers using GPU-accelarated framework [deeplearn.js](https://github.com/PAIR-code/deeplearnjs)

You can repeat the whole experiment in 3 steps:
- Run train.&#8203;py to get ./weights folder
- Build generator.ts 
- Run node server and open index.html
- You are awesome!


##### ./model contents:
-  train.&#8203;py - creates and trains the model, saves generator weights
-  GAN.&#8203;py - model definition (originated from [this repo](https://github.com/hwalsuklee/tensorflow-generative-model-collections))
-  yellowfin.&#8203;py - powerful custom TF optimizer, [original repo](https://github.com/JianGoForIt/YellowFin). Not really required, just a nice thing. 
-  ops.&#8203;py and utils.&#8203;py  - simplified TF operations and other utility functions

##### ./browser contents:
- generator.ts - downloads weights and defines generator model
- index.html - simple wrapper
- other configs
