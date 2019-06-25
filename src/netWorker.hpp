#ifndef net_thr_hpp
#include <string>
class netWorker {
    public:
        netWorker(string* url);
        void run(int sid);        
    private:
        string* url;
}
#endif