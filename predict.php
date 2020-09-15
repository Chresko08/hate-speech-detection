<?php
//Receive POST Parameters

$tweets=$_POST["tweets"];
system("location_of installed_python3_cmd_launcher testing_model.py ".$tweets." 2>&1");

?>
