## #include <ext\pipeline.h>

## Host Side
Prototype | Description | Tags
--- | --- | :---:
```typedef FDTYPE``` | OS type of fd.
```typedef PIDTYPE``` | OS type of pid.
```#define __BAD_FD``` | OS bad fd.
```#define __BAD_FD``` | OS bad pid.
```int pipelineCleanup(int numPids, PIDTYPE *pids, int child_siginfo);``` | Cleanup the pipeline's children.
```int pipelineCreate(int argc, char **argv, PIDTYPE **pidsPtr, FDTYPE *inPipePtr, FDTYPE *outPipePtr, FDTYPE *errFilePtr, FDTYPE process, pipelineRedir *redirs);``` | Creates the pipeline.
```void pipelineOpen(pipelineRedir &redir);``` | Ack the pipeline.
```void pipelineClose(pipelineRedir &redir);``` | Ack the pipeline.
```void pipelineRead(pipelineRedir &redir);``` | Reads from the pipeline.