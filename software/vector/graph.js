GRAPH = {};

GRAPH.Graph = function(canvasId, {dpi = 100, width = 4, height = 2, lineWidth = 2} = {}) {
    this.canvasElement = document.getElementById(canvasId);
    this.ctx = this.canvasElement.getContext("2d");

    this.canvasElement.width = dpi * width;
    this.canvasElement.height = dpi * height;
    this.canvasElement.style.width = width + "in";
    this.canvasElement.style.height = height + "in";
    this.ctx.imageSmoothingEnabled = true;

    this.lineWidth = lineWidth;
    this.width = dpi * width;
    this.height = dpi * height;

    //this.ctx.scale(dpi, dpi);
    this.drawGrid();
}
GRAPH.Graph.prototype.reset = function() {
    this.ctx.clearRect(0, 0, this.width, this.height);
}
GRAPH.Graph.prototype.drawLine = function(sx,sy,ex,ey) {
    this.ctx.lineWidth = this.lineWidth;
    this.ctx.beginPath();
    this.ctx.moveTo(sx, sy);
    this.ctx.lineTo(ex, ey);
    this.ctx.stroke();
}
GRAPH.Graph.prototype.drawGrid = function() {
    const stepX = this.width / 10;
    const stepY = this.height / 10;
    this.ctx.strokeStyle = "#ccc"
    for (let i=0; i<=10; i++) {
        this.drawLine(0,i*stepY,this.width,i*stepY);
        this.drawLine(i*stepX, 0, i*stepX, this.height);
    }
    this.ctx.strokeStyle = "#000"
    this.drawLine(0, this.height/2, this.width, this.height/2);
}
GRAPH.Graph.prototype.drawSampler = function(sampler) {
    let halfHeight = this.height / 2 - 1;
    let y = 1 + halfHeight + halfHeight * sampler.sample(0, 0);
    this.ctx.strokeStyle = "#00f"
    this.ctx.lineWidth = this.lineWidth;
    this.ctx.beginPath();
    this.ctx.moveTo(0, y);
    for (let i=1; i<=this.width; i++) {
        y = 1 + halfHeight + halfHeight * sampler.sample(i, i/this.width);
        this.ctx.lineTo(i, y);
    }
    this.ctx.stroke();
}