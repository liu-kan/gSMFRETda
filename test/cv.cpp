#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <unistd.h>
int two=2;
std::mutex *pm=new std::mutex[two];
std::mutex *pm1=new std::mutex[1];
std::condition_variable *pcv=new std::condition_variable[two] ;
std::condition_variable *pcv1=new std::condition_variable[1];
std::condition_variable &cv1=pcv[1];
std::condition_variable &cv=pcv[0];
std::mutex &m1=pm[1];
std::mutex &m=pm[0];
std::string data;
bool ready = false,ready1 = false;
bool processed = false;
/*
如果共用ready可能出现的情况：
    main set data date1, worker get date1 signal first, never get data signal
 */ 
void worker_thread()
{
    // 等待直至 main() 发送数据
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, []{return ready;});
 
    // 等待后，我们占有锁。
    std::cout << "Worker thread is processing data\n";
    data += " after processing";
    ready=false;
    lk.unlock();
    // std::this_thread::yield();
    // lk.lock();
    std::unique_lock<std::mutex> lk1(m1);
    // std::unique_lock<std::mutex> lk(m);
    cv1.wait(lk1, []{return ready1;});
 
    // 等待后，我们占有锁。
    std::cout << "Worker thread is processing data2\n";
    lk1.unlock();
    lk.lock();
    data += " after processing";

    // 发送数据回 main()
    processed = true;
    std::cout << "Worker thread signals data processing completed\n";
 
    // 通知前完成手动解锁，以避免等待线程才被唤醒就阻塞（细节见 notify_one ）
    lk.unlock();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    cv.notify_one();
}
 
int main()
{
    
 
    data = "Example data";
    // 发送数据到 worker 线程
    std::unique_lock<std::mutex> lk(m);
    // if(lk.try_lock())
    {
        // std::lock_guard<std::mutex> lk(m);
        // lk.lock();
        ready = true;
        std::cout << "main() signals data ready for processing\n";
        // lk.unlock();
    }
    cv.notify_one();
    
    
    
    // std::this_thread::yield(); 
    std::unique_lock<std::mutex> lk1(m1);
    {
        // lk1.lock();
        ready1 = true;
        std::cout << "main() signals data ready for processing 2\n";
        lk1.unlock();
    }
    
    cv1.notify_one();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::thread worker(worker_thread);
    // 等候 worker
    {
        std::unique_lock<std::mutex> lk(m);
        cv.wait(lk, []{return processed;});
    }
    std::cout << "Back in main(), data = " << data << '\n';
 
    worker.join();
}