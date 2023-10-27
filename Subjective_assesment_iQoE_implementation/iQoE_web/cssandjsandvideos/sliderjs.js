var slider = document.getElementById("myRange");
var output = document.getElementById("demo");
output.innerHTML = slider.value; // Display the default slider value

// Update the current slider value (each time you drag the slider handle)
slider.oninput = function() {
  output.innerHTML = this.value;
}

function confirm(){
    var score = output.innerHTML;
    document.getElementById("var1").value=score;
    document.getElementById("myForm").submit();
}

function replay(){
    window.location.href = '../show_video_replay';
}

function play_reference(){
    window.location.href = '../show_video_reference_score_window';
}