#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <nanomsg/nn.h>
#include <nanomsg/pair.h>

#define NODE0 "node0"
#define NODE1 "node1"

void
fatal(const char *func)
{
        fprintf(stderr, "%s: %s\n", func, nn_strerror(nn_errno()));
        exit(1);
}

int
send_name(int sock, const char *name, const char *url)
{
        printf("%s: SENDING \"%s\" at %s\n", name, name,url);
        int sz_n = strlen(name) + 1; // '\0' too
        return (nn_send(sock, name, sz_n, 0));
}

int
recv_name(int sock, const char *name, const char *url)
{
        char *buf = NULL;
        int result = nn_recv(sock, &buf, NN_MSG, 0);
        if (result > 0) {
                printf("%s: RECEIVED \"%s\" at %s\n", name, buf,url); 
                nn_freemsg(buf);
        }
        return (result);
}

int
send_recv(int sock, const char *name, const char *url)
{
        int to = 100;
        // if (nn_setsockopt(sock, NN_SOL_SOCKET, NN_RCVTIMEO, &to,
        //     sizeof (to)) < 0) {
        //         fatal("nn_setsockopt");
        // }
        int c=0;
        while (c<2) {
                recv_name(sock, name,url);
                sleep(1);
                // send_name(sock, name,url);
                // sleep(1);
                // send_name(sock, name,url);
                // sleep(1);
                // recv_name(sock, name,url);                
                c++;
        }
}

int
node0(const char *url)
{
        int sock;
        if ((sock = nn_socket(AF_SP, NN_PAIR)) < 0) {
                fatal("nn_socket");
        }
         if (nn_bind(sock, url) < 0) {
                fatal("nn_bind");
        }
        return (send_recv(sock, NODE0,url));
}

int
node1(const char *url)
{
        int sock;
        if ((sock = nn_socket(AF_SP, NN_PAIR)) < 0) {
                fatal("nn_socket");
        }
        if (nn_connect(sock, url) < 0) {
                fatal("nn_connect");
        }
        return (send_recv(sock, NODE1,url));
}

int
main(const int argc, const char **argv)
{
        if ((argc > 1) && (strcmp(NODE0, argv[1]) == 0)){
                (node0(argv[2]));
                if (argc > 3)
                (node0(argv[3]));
        }
                

        if ((argc > 1) && (strcmp(NODE1, argv[1]) == 0)){
                (node1(argv[2]));
                if (argc > 3)
                (node1(argv[3]));
        }

        fprintf(stderr, "Usage: pair %s|%s <URL> <ARG> ...\n", NODE0, NODE1);
        return 1;
}


/*

gcc pair.c -lnanomsg -o pair

./pair node0 ipc:///tmp/pair.ipc ipc:///tmp/pair1.ipc & node0=$!
./pair node1 ipc:///tmp/pair.ipc ipc:///tmp/pair1.ipc & node1=$!
sleep 12
kill $node0 $node1

*/