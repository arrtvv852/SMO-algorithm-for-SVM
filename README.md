# SMO-algorithm-for-SVM
SMO-SVM algorithm to predict alphabet DU data.
## Preporcessing
First, change the directory to the directory you put the training and testing file. <br />
![image](/readme_img/p01.png)<br />
Also, change the file name to the file name of your training and testing file. <br />
![image](/readme_img/p02.png)<br />
## Parameter setting
Run the python program and you will see these: <br />
![image](/readme_img/p03.png)<br />
Choose the kernel type you want.
Then choose the tolerance value:<br />
![image](/readme_img/p04.png)<br />
Key in the epsilon value(0.01 suggested).
Then key in Y or N(do cross validation or not)<br />
![image](/readme_img/p05.png)<br />
If you choose Y, then choose the n value you want to do cross validation (5 or 10 suggested).<br />
![image](/readme_img/p06.png)<br />
## Progress
If you choose to do cross validation, then you will see these: <br />
![image](/readme_img/p07.png)<br />
Please be patient and keep waiting. <br />
## Cross validating
If you choose to do the cross validation, then your will see the risk you each C value you tried as following. And you can check if the optimal C value is chosen by here. <br />
![image](/readme_img/p08.png)<br />
## Output
If you have done all the cross validation progress, the following will be shown. Then you can choose whether to see the result of exit directly (1 or 2 or 3).<br />
![image](/readme_img/p09.png)<br />
You can choose to see the result then the result will be printed as following: <br />
![image](/readme_img/p10.png)<br />
3 is for saving detail result data.
