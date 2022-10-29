//
// Created by eliseysudakov on 10/2/22.
//

#include "courier.h"

void to_json(nlohmann::json &j, const Courier &courier) {
    j = {
            {"id", courier.id}
    };
}