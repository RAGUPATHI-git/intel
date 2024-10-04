let moon = document.getElementById("moon");
let text = document.getElementById("text");
let train = document.getElementById("train");
let desert_moon = document.getElementById("desert-moon");
let man = document.getElementById("man");


window.addEventListener("scroll",()=>{
    let value = window.scrollY;
    moon.style.top = value * .9 + "px";
    text.style.top = 80 + value * -0.2 + '%';
    train.style.left = value*1.5 +'px';
    desert_moon.style.top = value *.3 +'px';
    man.style.left = value* .6 + 'px';
})

//progress bar



let calcScrollValue = ()=>{
    let scrollprogress = document.getElementById("progress");
    let pos = document.documentElement.scrollTop;
    let calcHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
    let scrollvalue = Math.round((pos*100)/calcHeight);


    if(pos > 100){
        scrollprogress.style.display = "grid";
    }
    else{
        scrollprogress.style.display = "none";
    }
  

    scrollprogress.addEventListener("click",()=>{
        document.documentElement.scrollTop =0;
    });

    scrollprogress.style.background = "conic-gradient(#194eb9" + scrollvalue +"," +"#67ccff" +scrollvalue +")";

};



window.onscroll = calcScrollValue;
window.onload  = calcScrollValue;