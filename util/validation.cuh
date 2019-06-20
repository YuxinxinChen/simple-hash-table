#ifndef ERROR_UTIL
#include "error_util.cuh"
#endif

#include <vector>
#include <queue> 

using namespace std;

namespace host {

    template <typename VertexId, typename SizeT, typename Value, typename Rank>
    void PrValid(Csr<VertexId, SizeT, Value> &csr, Csr<VertexId, SizeT, Value> &csc, Pr<VertexId, SizeT, Value, Rank> &pr, int my_pe)
    {
        float lambda = pr.lambda;
        float epsilon = pr.epsilon;
    
        SizeT edges = csr.edges;
        VertexId nodes = csr.nodes;
    
        Rank *h_rank = (Rank *)malloc(sizeof(Rank)*nodes);
        MALLOC_CHECK(h_rank);
        Rank *h_res = (Rank *)malloc(sizeof(Rank)*nodes);
        MALLOC_CHECK(h_res);
        memset(h_res, 0, sizeof(Rank)*nodes);
    
        queue<VertexId> wl;
        for(VertexId i=0; i<nodes; i++)
        {
            h_rank[i] = 1.0-lambda;
            wl.push(i);
    
            VertexId sourceStart = csc.row_offset[i];
            VertexId sourceEnd= csc.row_offset[i+1];
            for(int j=0; j<sourceEnd-sourceStart; j++)
            {
                VertexId sourceId = csc.column_indices[sourceStart+j];
    //            h_res[i] = h_res[i] + 1.0/(csr.row_offset[sourceId+1]-csr.row_offset[sourceId]);
                h_res[i] = h_res[i] + (1.0-lambda)*lambda/(csr.row_offset[sourceId+1]-csr.row_offset[sourceId]);
            }
//            h_res[i] = (1.0-lambda)*lambda*h_res[i];
        } //for vertices
        //finish res and rank init

        while(!wl.empty())
        {
            VertexId node_item = wl.front();
            wl.pop();
            h_rank[node_item] = h_rank[node_item]+h_res[node_item];
            VertexId destStart = csr.row_offset[node_item];
            VertexId destEnd = csr.row_offset[node_item+1];
            Rank res_owner = h_res[node_item];
            for(int j=0; j<destEnd-destStart; j++)
            {
                VertexId dest_item = csr.column_indices[j+destStart];
                Rank res_old = h_res[dest_item];
                h_res[dest_item] = h_res[dest_item] + res_owner*lambda/(Rank)(destEnd-destStart); 
                if(res_old < epsilon && h_res[dest_item] > epsilon)
                    wl.push(dest_item);
            }
            h_res[node_item] = 0.0;
    
        }//while worklist
    
        Rank totalRank=0.0;
        Rank totalRes = 0.0;
        for(VertexId i=0; i<nodes; i++)
        {
            totalRank = totalRank + h_rank[i];
            totalRes= totalRes+ h_res[i];
        }
        
     //   for(int i=0; i<nodes; i++)
     //       cout << h_rank[i] << " ";
     //   cout << endl;

        cout << "CPU total mass: " << totalRank + totalRes/(1.0-lambda) << " CPU total res: " << totalRes << " CPU total rank: " << totalRank << endl;
    
        float error=0.0;
        Rank *check_rank = (Rank *)malloc(sizeof(Rank)*pr.nodes);
        Rank *check_res = (Rank *)malloc(sizeof(Rank)*pr.totalNodes);
        MALLOC_CHECK(check_rank);
        MALLOC_CHECK(check_res);
        CUDA_CHECK(cudaMemcpy(check_rank, pr.d_rank, sizeof(Rank)*pr.nodes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(check_res, pr.d_res, sizeof(Rank)*pr.totalNodes, cudaMemcpyDeviceToHost));
        Rank sum_rank=0.0;
        Rank sum_res = 0.0;
        cout << "GPU rank:\n";
        for(VertexId i=0; i<pr.nodes; i++)
        {
     //       error = error + abs(check_rank[i]-h_rank[i]/totalRank);
            error = error + abs(check_rank[i]-h_rank[i+pr.startNode]);
     //       cout << check_rank[i] << " ";
            sum_rank = sum_rank + check_rank[i];
        }
        cout << endl;
        cout << "GPU res:\n";
        for(VertexId i=0; i<pr.totalNodes; i++)
        {
     //       cout << check_res[i] << " ";
            sum_res = sum_res + check_res[i];
        }
        cout << endl;

        cout<<"PE" << pr.my_pe << " accumulated error: "<< error << " GPU sum_rank: "<< sum_rank << " GPU sum_res: "<< sum_res << " GPU total mass: "<< sum_rank+sum_res/(1.0-lambda) << endl;
        if(error > 0.01) cout << "FAILE\n";

  //     Rank *check_res = (Rank *)malloc(sizeof(Rank)*nodes);
  //      MALLOC_CHECK(check_res);
  //      CUDA_CHECK(cudaMemcpy(check_res, pr.d_res, sizeof(Rank)*nodes, cudaMemcpyDeviceToHost));
  //      float error=0.0;
  //      for(int i=pr.startNode; i<=pr.endNode; i++)
  //          error = error + abs(check_res[i]-h_res[i]);
  //      cout <<"pe: "<< my_pe << " error :" << error << endl;
  //   //   cout << "Print the first 20 res: \n";
  //   //   cout << "host:\n";
  //   //   for(int i=0; i<20; i++)
  //   //       cout << h_res[i] << " ";
  //   //   cout << endl;
  //   //   cout << "device:\n";
  //   //   for(int i=0; i<20; i++)
  //   //       cout << check_res[i] << " ";
  //   //   cout << endl;
  //      CUDA_CHECK(cudaMemcpy(check_res, pr.d_rank, sizeof(Rank)*pr.nodes, cudaMemcpyDeviceToHost));
  //      cout << "Print rank\n";
  //      for(int i=0; i<pr.nodes; i++)
  //          cout << check_res[i] << " ";
  //      cout << endl;
    }//PrValid

} //namespace
