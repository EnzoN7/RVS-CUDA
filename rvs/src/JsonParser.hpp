/* The copyright in this software is being made available under the BSD
* License, included below. This software may be subject to other third party
* and contributor rights, including patent rights, and no such rights are
* granted under this license.
*
* Copyright (c) 2010-2018, ITU/ISO/IEC
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
*  * Redistributions of source code must retain the above copyright notice,
*    this list of conditions and the following disclaimer.
*  * Redistributions in binary form must reproduce the above copyright notice,
*    this list of conditions and the following disclaimer in the documentation
*    and/or other materials provided with the distribution.
*  * Neither the name of the ITU/ISO/IEC nor the names of its contributors may
*    be used to endorse or promote products derived from this software without
*    specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
* THE POSSIBILITY OF SUCH DAMAGE.
*/

/*
Original authors:

Universite Libre de Bruxelles, Brussels, Belgium:
  Sarah Fachada, Sarah.Fernandes.Pinto.Fachada@ulb.ac.be  Daniele Bonatto, Daniele.Bonatto@ulb.ac.be  Arnaud Schenkel, arnaud.schenkel@ulb.ac.be

Koninklijke Philips N.V., Eindhoven, The Netherlands:
  Bart Kroon, bart.kroon@philips.com
  Bart Sonneveldt, bart.sonneveldt@philips.com
*/

#ifndef _JSON_PARSER_HPP_
#define _JSON_PARSER_HPP_

#include <iosfwd>
#include <string>
#include <memory>

namespace json
{
	struct Value;
	struct Object;

	class Node
	{
	public:
		enum class Type
		{
			number,
			string,
			array,
			object,
			boolean,
			null
		};

		/** Read JSON from an input stream */
		static Node readFrom(std::istream&);

		/* Specify overrides (for Object only) */
		void setOverrides(Node overrides);

		/** Query the value type of this node */
		Type type() const;

		/** Query an object parameter, Null if not found
		
		It is possible to specify overrides */
		Node optional(std::string const& key) const;

		/** Query an object parameter, throws "parameter KEY is required" unless found 
		
		It is possible to specify overrides */
		Node require(std::string const& key) const;

		/** Query an array by index */
		Node at(std::size_t index) const;

		/** Query the size of an array */
		std::size_t size() const;

		/** Query the value of a number */
		double asDouble() const;

		/** Query the value of a number */
		float asFloat() const;

		/** Query the value of a number and require it to be an integer */
		int asInt() const;

		/** Query the value of a string */
		std::string const& asString() const;

		/** Query the value of a boolean */
		bool asBool() const;

		/** Anything apart from false and null is true */
		operator bool() const;

	private:
		Node();
		Node(std::shared_ptr<Value>);

		std::shared_ptr<Value> m_value;
		std::shared_ptr<Object> m_overrides;
	};	
}

#endif