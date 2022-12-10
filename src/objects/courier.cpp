//
// Created by eliseysudakov on 10/2/22.
//

#include "courier.h"
#include <iostream>

void to_json(nlohmann::json &j, const Courier &courier) {
    j = {
            {"id", courier.id}
    };
}

void from_json(const nlohmann::json &j, Courier &courier) {
    j.at("id").get_to(courier.id);
}