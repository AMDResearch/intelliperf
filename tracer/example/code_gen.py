def generate_header(args: list[str]) -> str:
    header_path = "/tmp/KernelArguments.hpp"
    member_names = [arg.split()[-1] for arg in args]
    members = ";\n    ".join(args) + ";"
    as_tuple_members = ", ".join(member_names)

    header_content = f"""#pragma once
#include <tuple>
struct KernelArguments {{
    {members}

    auto as_tuple() const {{
        return std::tie({as_tuple_members});
    }}
}};
"""
    with open(header_path, "w") as header_file:
        header_file.write(header_content)
    return header_path
