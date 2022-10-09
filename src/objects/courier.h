//
// Created by eliseysudakov on 10/2/22.
//

#pragma once

#include <string>

class Courier {
public:
    Courier(std::string id) :
            id(std::move(id)) {}

    std::string id;
};
