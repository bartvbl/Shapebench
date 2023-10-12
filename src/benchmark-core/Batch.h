#pragma once

#include <algorithm>

namespace Shapebench {
    template<typename indexType>
    class Batch {
        indexType totalBatchCount = 0;
        indexType totalElementCount = 0;
        indexType elementsPerBatch = 0;
        indexType currentBatchIndex = 0;
        indexType currentElementIndex = 0;
        indexType currentBatchSize = 0;
        indexType currentBatchElementStartIndex = 0;
        indexType currentBatchElementEndIndex = 0;

        void computeBatchBounds() {
            currentBatchElementStartIndex = currentBatchIndex * elementsPerBatch;
            currentBatchElementEndIndex = std::min((currentBatchIndex + 1) * elementsPerBatch, totalElementCount);
            currentBatchSize = currentBatchElementEndIndex - currentBatchElementStartIndex;
        }

    public:
        Batch(indexType count, indexType batchSize) {
            totalElementCount = count;
            totalBatchCount = (count / batchSize) + (count % batchSize > 0 ? 1 : 0);
            elementsPerBatch = std::min(batchSize, count);
            computeBatchBounds();
        }

        indexType batchSize() {
            return currentBatchSize;
        }
        indexType elementIndex() {
            return currentElementIndex;
        }
        indexType next() {
            currentElementIndex++;
            if(currentElementIndex == currentBatchElementEndIndex) {
                
            }
        }

        bool batchComplete() {
            currentElementIndex >= currentBatchElementEndIndex;
        }
        void nextElement() {
            currentElementIndex++;
        }

    };
}
