#pragma once

#include <algorithm>

namespace ShapeBench {
    template<typename indexType>
    class Batch {
        indexType totalElementCount = 0;
        indexType elementsPerBatch = 0;
        indexType currentElementIndex = 0;
        indexType currentBatchSize = 0;

        void computeBatchSize() {
            indexType currentBatchElementStartIndex = currentElementIndex;
            indexType currentBatchElementEndIndex = std::min(currentElementIndex + elementsPerBatch, totalElementCount);
            currentBatchSize = currentBatchElementEndIndex - currentBatchElementStartIndex;
        }

    public:
        Batch(indexType count, indexType batchSize) {
            totalElementCount = count;
            elementsPerBatch = std::min(batchSize, count);
            computeBatchSize();
        }

        indexType batchSize() {
            return currentBatchSize;
        }
        indexType next() {
            currentElementIndex++;
            if(isNewBatch()) {
                computeBatchSize();
            }
            return currentElementIndex;
        }
        bool isNewBatch() {
            return currentElementIndex % currentBatchSize == 0;
        }

    };
}
