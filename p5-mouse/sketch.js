var r_array = [0.0];
var ang_array = [0.0];
var frame_array = [0.0];
var ms_array = [0.0];

var fps = 60;

var ms_offset = 2000;

var session_time = 7*60; //seconds
// var capturer = new CCapture({
//   format: 'jpg',
//   name: 'name',
//   framerate: 60,
//   workersPath: './',
//   verbose: true
// });

var canvas;
var endSession = false;

class Tail {
  constructor(num_elements) {
    this.num = num_elements;
    this.mx = new Array(this.num).fill(0);
    this.my = new Array(this.num).fill(0);
    this.which = this.num;
  }

  update(x, y, f_count) {
    if(f_count < frameRate()) {
      this.mx.fill(x);
      this.my.fill(y);
    } else {
      this.which = f_count % this.num;
      this.mx[this.which] = x;
      this.my[this.which] = y;
    }
  }

  draw () {
    noFill();
    stroke(1);
    beginShape();
    for (let i = 0; i < this.num; i++) {
      let index = (this.which + 1 + i) % this.num;
      curveVertex(this.mx[index], this.my[index]);
    }
    endShape();
    fill(51);
  }
}

function setup() {

  var p5Canvas = createCanvas(windowWidth, windowHeight);
  canvas = p5Canvas.canvas;

  frameRate(fps);
  mouseTail = new Tail(15);
  // capturer.start();
}

function draw() {
  background(204);
  let elapsed_ms = millis();
  calcRelativePolarCoordinates(mouseX, mouseY, pmouseX, pmouseY, frameCount, elapsed_ms);
  mouseTail.update(mouseX, mouseY, frameCount);
  mouseTail.draw();

  textAlign(CENTER);
  textSize(20);

  if(elapsed_ms > session_time*1000 && endSession == false) {
    downloadData();
    endSession = true;
  }
  if(elapsed_ms > ms_offset && endSession == false) {
    let elapsed_s = floor((elapsed_ms-ms_offset)/1000);
    let elapsed_m = floor(elapsed_s/60);
    text(nf(elapsed_m, 2, 0) + ':' + nf(elapsed_s % 60, 2, 0) + ':' +nf((elapsed_ms-ms_offset)%1000, 3, 0), windowWidth-100, 50);
  }


    // capturer.capture(canvas);
}

function windowResized() {
  resizeCanvas(windowWidth, windowHeight);
}

function calcRelativePolarCoordinates(x, y, pX, pY, frame, ms) {
  let relP = [x-pX, y-pY];

  let r = Math.sqrt(Math.pow(relP[0], 2) + Math.pow(relP[1], 2));
  let ang = ((relP[0] == 0) ? 0 : Math.atan( relP[1] / relP[0] ));
  //console.log(r, ang*(180/Math.PI));
  r_array.push(r);
  ang_array.push(ang);
  frame_array.push(frame);
  ms_array.push(ms);
}

function downloadData() {
  var table = new p5.Table();
  table.addColumn('frames');
  table.addColumn('millis');
  table.addColumn('radial');
  table.addColumn('angular');

  for (let i = 0; i < frame_array.length; i++) {
    let newRow = table.addRow();
    newRow.setNum('frames', frame_array[i]);
    newRow.setNum('millis', ms_array[i]);
    newRow.setNum('radial', r_array[i]);
    newRow.setNum('angular', ang_array[i]);
  }

  saveTable(table, "relative_polar_coordinates.csv");
}

function keyTyped() {
  if (key === 'p') {

    downloadData();

    // capturer.stop();
    // capturer.save();
  }
}
