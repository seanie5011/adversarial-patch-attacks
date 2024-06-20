# Training pictures

Contains pictures and videos of the patch over every epoch. Not all are saved for storage purposes.

## videos

Videos were merged by following (requires ffmpeg): 

1. Create file `mylist.txt` where your files are and add the files you wish to merge, like:
```
file 'myfile.mp4'
file 'anotherfile.mp4'
```
2. Use ffmpeg in a command prompt where the files are and type the following (new file will be called `mergedfile.mp4`):
```
ffmpeg -f concat -i mylist.txt -c copy mergedfile.mp4
```

## run 1

50 epochs, 102m 26s, 100%, lr1, prob0.9, noise0.1, target0, batchsize1, numworkers4, iterations1000

## run 2

25 epochs, 55m 026s, 100%, lr1, prob0.9, noise0.1, target1, batchsize1, numworkers4, iterations1000