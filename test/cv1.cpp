#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>
 #include <unistd.h>
std::mutex m,m1;
std::condition_variable cv,cv1;
std::string data;
bool ready = false,ready1 = false;
bool processed = false;
/*
如果始终使用问答形式，并保持顺序，
可共用ready
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
    // 发送数据回 main()
    processed = true;
    std::cout << "Worker thread signals data processing completed\n";
 
    // 通知前完成手动解锁，以避免等待线程才被唤醒就阻塞（细节见 notify_one ）
    lk.unlock();
    cv.notify_one();

    lk.lock();
    cv.wait(lk, []{return ready;});
    // 等待后，我们占有锁。
    std::cout << "Worker thread is processing data2\n";
    data += " after processing";
    ready=false;
    // 发送数据回 main()
    processed = true;
    std::cout << "Worker thread signals data processing completed\n";
 
    // 通知前完成手动解锁，以避免等待线程才被唤醒就阻塞（细节见 notify_one ）
    lk.unlock();
    cv.notify_one();    
}
 
int main()
{
    std::thread worker(worker_thread);
 
    data = "Example data";
    // 发送数据到 worker 线程
    std::unique_lock<std::mutex> lk(m);
    // if(lk.try_lock())
    {
        // std::lock_guard<std::mutex> lk(m);
        // lk.lock();
        ready = true;
        std::cout << "main() signals data ready for processing\n";
        lk.unlock();
    }
    cv.notify_one();
    // 等候 worker
    {
        // std::unique_lock<std::mutex> lk(m);
        lk.lock();
        cv.wait(lk, []{return processed;});
    }
    std::cout << "Back in main(), data = " << data << '\n';
    processed=false;
    lk.unlock();
/////////////////////
    // if(lk.try_lock())
    {
        // std::lock_guard<std::mutex> lk(m);
        lk.lock();
        ready = true;
        std::cout << "main() signals data ready for processing2\n";
        lk.unlock();
    }
    cv.notify_one();
    // 等候 worker
    {
        lk.lock();
        cv.wait(lk, []{return processed;});
    }
    std::cout << "Back in main(), data = " << data << '\n';
    lk.unlock();


 
    worker.join();
}