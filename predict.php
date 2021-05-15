<?php
//Receive POST Parameters
if (isset($_POST['predict']))
{
    $tweet=$_POST["tweets"];
    echo system("C:\Users\shubh\AppData\Local\Programs\Python\Python37\python.exe C:\inetpub\wwwroot\hate-speech-detection\testing_model.py $tweet");
}
?>