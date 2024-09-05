<?php
//Receive POST Parameters
if (isset($_POST['tweet'])){

		$tweets=$_POST['tweet'];
		$ti="\"".$tweets."\"";
		$ti="You are bloody idiot";
		system("python3 Test.py ".$ti." 2>&1");
		echo "Hello";
}

?>
