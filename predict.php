<?php
//Receive POST Parameters

$tweets=$_POST["tweets"];
system("/usr/anaconda/bin/python3 Test.py ".$tweets." 2>&1");

?>
