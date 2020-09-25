# TODO list

## 2020-9-18 added
* Allow user to choose 64bit or 32bit precision

## 2019-7-10 added
* ~~try multi streamWorker workers~~
```bash
git branch multiThreadMultiStream e3cf1b4a92071e561ceed992cb18147915fd8f20 
```
* ~~Test std::timed_mutex::try_lock_for vs std::mutex::try_lock~~

## 2019-6-18 added
* ~~Use cudaStreamQuery to Query all stream, or cudaStreamAddCallback~~

## 2019-3-11 added
* ~~Use streams in GPU.~~
* ~~Run cudaOccupancyMaxPotentialBlockSize 1 time to get blockSize.~~
* Use shared memory in kernel to reduce the usage of reg in kernel thread.