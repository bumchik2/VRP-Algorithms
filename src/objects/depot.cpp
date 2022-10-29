//
// Created by eliseysudakov on 10/2/22.
//

#include "depot.h"

void to_json(nlohmann::json &j, const Depot &depot) {
    j = {
            {"id",    depot.id},
            {"point", {
                              {"lat", depot.lat},
                              {"lon", depot.lon},
                      }}
    };
}
